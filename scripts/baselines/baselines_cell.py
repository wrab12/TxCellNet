"""Cell-level baselines, all panel-matched: for every panel group (results/bench/panels.json) each baseline is trained on
kidney cells reading only the group's genes (kidney genes measured in both sections and in every member panel; counts
normalised over those same genes), then applied to the group's cases. Same cells, markers and metrics as run_bench_gen.py.

  ridge_pm      ridge regression (lambda 10) on standardised log-normalised counts
  mlp_pm        2-layer MLP (512-256, GELU, dropout .2), 12 epochs
  celltype_pm   cell-type lookup: k-means (30 clusters) in PCA-50 space of the kidney cells, predict the cluster's mean protein
  knn_pm        reference mapping: mean protein of the 50 nearest kidney cells in PCA-50 space
  nichetrans_pm NicheTrans (official model class, 12 spatial neighbours, short training); not used in the paper, whose
                NicheTrans follows the official tutorial (nichetrans_official.py)
Training cells = train stripes of both kidney sections (held-out stripes are never seen, so kidney rows are fair).
Targets: 10x per-cell protein values, log1p, z-scored per section.
SEED (default 1): seed of the MLP / NicheTrans training (suffix _s<seed> for seeds other than 1).
GENESET=both (default): genes measured in both kidney sections; GENESET=union: all 495 vocabulary genes of the group,
zero for the cells of the section that lacks a gene (what TxCellNet reads). Method names get suffix '_u' for union.
Cross-section tier (kidney): KID_TRAIN=rcc|ccrcc trains on that section's training stripes only; CASES=a,b restricts the
predicted cases.
usage: [GENESET=both|union] baselines_cell.py OUT_DIR GROUP[,GROUP...]|all  (cases = the groups' prepared members)
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, h5py, torch, torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans

from txcellnet.bench import PROTS, combine, log
from txcellnet.metrics import metrics

sys.path.insert(0, os.environ.get('NICHETRANS_DIR', 'external/NicheTrans'))
from model.nicheTrans import NicheTrans

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
PANELS = json.load(open('results/bench/panels.json'))
GROUPS = list(PANELS) if sys.argv[2] == 'all' else sys.argv[2].split(',')
dev = 'cuda'; K_NB = 12


def featurize(X, genes, use):
    """cells x genes counts (csr) -> dense log1p(normalize_total 1e3) over the genes `use` (missing -> 0)."""
    pos = {g: i for i, g in enumerate(genes)}
    F_ = np.zeros((X.shape[0], len(use)), np.float32)
    have = [j for j, g in enumerate(use) if g in pos]
    A = X[:, [pos[use[j]] for j in have]].tocsr().astype(np.float32)
    tot = np.asarray(A.sum(1)).ravel()
    F_[:, have] = np.log1p((sp.diags(1e3 / np.maximum(tot, 1)) @ A).toarray())
    return F_


def knn_xy(xy, k=K_NB):
    return NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]


# ---------------- kidney reference ----------------
KID = []
for s in ['rcc', 'ccrcc']:
    with h5py.File(f'data/xenium_protein_{s}/outs/cell_feature_matrix.h5', 'r') as f:
        m = f['matrix']
        bc = np.array([b.decode() for b in m['barcodes'][:]]); ft = np.array([t.decode() for t in m['features']['feature_type'][:]])
        nm = np.array([t.decode() for t in m['features']['name'][:]])
        X = sp.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=tuple(m['shape'][:])).T.tocsr()
    G = X[:, ft == 'Gene Expression']; gn = list(nm[ft == 'Gene Expression'])
    P = np.log1p(X[:, ft == 'Protein Expression'].toarray().astype(np.float64)); pn = list(nm[ft == 'Protein Expression'])
    P = P[:, [pn.index(p) for p in PROTS]]; P = ((P - P.mean(0)) / (P.std(0) + 1e-6)).astype(np.float32)
    cells = pd.read_parquet(f'data/xenium_protein_{s}/outs/cells.parquet').set_index('cell_id').reindex(bc)
    xy = cells[['x_centroid', 'y_centroid']].values
    edges = np.array(json.load(open(f'results/cell_baseline/{s}_stripes.json'))['x_edges_um'])
    tr = (np.digitize(xy[:, 0], edges) % 2 == 0) & (np.asarray(G.sum(1)).ravel() > 0)
    if os.environ.get('KID_TRAIN') and s != os.environ['KID_TRAIN']: tr[:] = False  # cross-section tier
    KID.append(dict(G=G.tocsr(), genes=gn, P=P, nb=knn_xy(xy), tr=tr))
    log('kidney', s, 'cells', len(bc), 'train', int(tr.sum()))
BOTH = set(KID[0]['genes']) & set(KID[1]['genes'])
GSET = os.environ.get('GENESET', 'both'); SEED = int(os.environ.get('SEED', 1))
SUF = ('_u' if GSET == 'union' else '') + (f'_s{SEED}' if SEED != 1 else '')


class MLP(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(.2), nn.Linear(512, 256), nn.GELU(), nn.Dropout(.2), nn.Linear(256, len(PROTS)))

    def forward(s, x): return s.f(x)


def batched(fn, n, bs=8192):
    with torch.no_grad():
        return np.vstack([fn(torch.arange(i, min(i + bs, n), device=dev)) for i in range(0, n, bs)])


def train_group(gname):
    genes = [g for g in PANELS[gname]['genes'] if GSET == 'union' or g in BOTH]
    Fs = [featurize(k['G'], k['genes'], genes) for k in KID]
    off = np.cumsum([0] + [len(f) for f in Fs])
    Fk = np.vstack(Fs); Pk = np.vstack([k['P'] for k in KID]); TR = np.concatenate([k['tr'] for k in KID])
    NB = np.vstack([k['nb'] + o for k, o in zip(KID, off[:-1])])
    tr = np.flatnonzero(TR)
    mu, sd = Fk[tr].mean(0), Fk[tr].std(0) + 1e-6
    Z = (Fk - mu) / sd
    M = dict(genes=genes, mu=mu, sd=sd)
    # ridge
    Zt, Pt = Z[tr].astype(np.float64), Pk[tr].astype(np.float64)
    M['ridge_W'] = np.linalg.solve(Zt.T @ Zt + 10.0 * np.eye(Zt.shape[1]), Zt.T @ (Pt - Pt.mean(0))); M['ridge_b'] = Pt.mean(0)
    # PCA-50 (kidney train cells) for the lookup baselines
    U, S_, Vt = np.linalg.svd(Zt[np.random.default_rng(0).choice(len(tr), min(len(tr), 200_000), replace=False)], full_matrices=False)
    M['pca'] = Vt[:50].T.astype(np.float32)
    E = (Z[tr] @ M['pca']).astype(np.float32)
    km = MiniBatchKMeans(30, random_state=0, batch_size=8192, n_init=3).fit(E)
    M['km_C'] = km.cluster_centers_.astype(np.float32)
    M['km_P'] = np.stack([Pk[tr][km.labels_ == c].mean(0) for c in range(30)]).astype(np.float32)
    M['ref_E'], M['ref_P'] = E, Pk[tr]
    # MLP
    torch.manual_seed(SEED)
    Zg, Pg = torch.from_numpy(Z.astype(np.float32)).to(dev), torch.from_numpy(Pk).to(dev)
    mlp = MLP(len(genes)).to(dev); opt = torch.optim.AdamW(mlp.parameters(), 1e-3, weight_decay=1e-4)
    EP = 12; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    for ep in range(EP):
        mlp.train(); perm = np.random.default_rng(ep + 100 * (SEED - 1)).permutation(tr)
        for i in range(0, len(perm) - 1023, 1024):
            b = torch.from_numpy(perm[i:i + 1024]).to(dev)
            loss = nn.functional.mse_loss(mlp(Zg[b]), Pg[b]); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    mlp.eval(); M['mlp'] = mlp
    # NicheTrans on the same genes (log-normalised features, not standardised, as in the official tutorial)
    torch.manual_seed(SEED)
    Ft = torch.from_numpy(Fk).to(dev); NBt = torch.from_numpy(NB).to(dev)
    nt = NicheTrans(source_length=len(genes), target_length=len(PROTS), noise_rate=0.2, dropout_rate=0.2).to(dev)
    opt = torch.optim.Adam(nt.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.1); t0 = time.time()
    for ep in range(8):
        nt.train(); perm = np.random.default_rng(ep + 100 * (SEED - 1)).permutation(tr); tot = 0
        for i in range(0, len(perm) - 1023, 1024):
            b = torch.from_numpy(perm[i:i + 1024]).to(dev)
            loss = nn.functional.mse_loss(nt(Ft[b], Ft[NBt[b]]), Pg[b]); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
    nt.eval(); M['nt'] = nt
    torch.save(nt.state_dict(), f'{OUT}/nichetrans_{gname}{SUF}.pt'); torch.save(mlp.state_dict(), f'{OUT}/mlp_{gname}{SUF}.pt')
    log(gname, 'genes', len(genes), 'trained', f'{time.time() - t0:.0f}s NicheTrans')
    # kidney held-out stripes, vs 10x values (reference numbers for the in-distribution setting)
    te = np.flatnonzero(~TR)
    pk = predict_all(M, Fk[te], NB=None, Fall=Fk, idx=te, NBall=NB)
    for k_, v in pk.items():
        log(gname, 'kidney held-out', k_, 'median Pearson', round(float(np.median([np.corrcoef(v[:, j], Pk[te, j])[0, 1] for j in range(len(PROTS))])), 3))
    return M


def predict_all(M, F_, NB, Fall=None, idx=None, NBall=None):
    """F_: features of the cells to score; NicheTrans needs neighbour features: either NB (indices into F_) or
    Fall/NBall/idx (neighbours of cells idx, indices into Fall)."""
    Z = ((F_ - M['mu']) / M['sd']).astype(np.float32)
    out = {'ridge_pm': (Z @ M['ridge_W'] + M['ridge_b']).astype(np.float32)}  # names get SUF below
    Zg = torch.from_numpy(Z).to(dev)
    out['mlp_pm'] = batched(lambda b: M['mlp'](Zg[b]).cpu().numpy(), len(Z))
    E = torch.from_numpy(Z @ M['pca']).to(dev); C = torch.from_numpy(M['km_C']).to(dev)
    out['celltype_pm'] = M['km_P'][batched(lambda b: torch.cdist(E[b], C).argmin(1, keepdim=True).cpu().numpy(), len(Z)).ravel()]
    R_ = torch.from_numpy(M['ref_E']).to(dev); RP = torch.from_numpy(M['ref_P']).to(dev)
    out['knn_pm'] = batched(lambda b: RP[torch.cdist(E[b], R_).topk(50, largest=False).indices].mean(1).cpu().numpy(), len(Z), bs=256)
    if NB is not None:
        Ft, NBt = torch.from_numpy(F_).to(dev), torch.from_numpy(NB).to(dev)
        out['nichetrans_pm'] = batched(lambda b: M['nt'](Ft[b], Ft[NBt[b]]).cpu().numpy(), len(F_))
    else:
        Ft, NBt, It = torch.from_numpy(Fall).to(dev), torch.from_numpy(NBall).to(dev), torch.from_numpy(idx).to(dev)
        out['nichetrans_pm'] = batched(lambda b: M['nt'](Ft[It[b]], Ft[NBt[It[b]]]).cpu().numpy(), len(idx))
    return {k + SUF: v for k, v in out.items()}


rows = []
for gname in GROUPS:
    only = [c for c in os.environ.get('CASES', '').split(',') if c]
    cases = [c for c in PANELS[gname]['cases'] if os.path.exists(f'data/cases/{c}/meta.json') and (not only or c in only)]
    if not cases: continue
    M = train_group(gname)
    for c in cases:
        cd = f'data/cases/{c}'; meta = json.load(open(f'{cd}/meta.json'))
        cells = pd.read_parquet(f'{cd}/cells.parquet')
        Xc = sp.load_npz(f'{cd}/counts.npz').tocsr(); cg = json.load(open(f'{cd}/counts_genes.json'))
        ok = (cells['area_px'].values >= 10) & (np.asarray(Xc.sum(1)).ravel() >= 1)
        if 'eval_mask' in meta: ok &= cells[meta['eval_mask']].values.astype(bool)
        F_ = featurize(Xc, cg, M['genes'])
        preds = predict_all(M, F_, knn_xy(cells[['x_um', 'y_um']].values))
        os.makedirs(f'{OUT}/preds', exist_ok=True)  # per-cell predictions (rows = cells.parquet) for image-level comparisons
        for mn, pc in preds.items(): np.save(f'{OUT}/preds/{c}__{mn}.npy', pc.astype(np.float16))
        for key, mk in meta['markers'].items():
            tt = cells[mk['truth']].values.astype(np.float64)
            if mk.get('log', True): tt = np.log1p(np.clip(tt - mk.get('offset', 0), 0, None))
            for mn, pc in preds.items():
                rows.append(dict(case=meta['dataset'], marker=key, method=mn, panel=gname, n_genes=len(M['genes']),
                                 **metrics(combine(pc, mk['components'], key)[ok], tt[ok]), n=int(ok.sum())))
        df = pd.DataFrame(rows); df.to_csv(f'{OUT}/bench_rows.csv', index=False)
        log(c, '\n' + df[df.case == meta['dataset']].groupby('method').spearman.median().round(3).to_string())
    del M; torch.cuda.empty_cache()
log('done')
