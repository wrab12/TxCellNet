"""DGAT (Wang et al., Nat Commun 2026; github.com/osmanbeyoglulab/DGAT, commit b0128f3) on our benchmark.

Official parts, unchanged: the model classes (Model/dgat.py: two GAT encoders, mRNA and per-protein decoders, hidden 1024,
dropout 0.3), the loss (5 RNA reconstruction + protein reconstruction + latent alignment + 3 protein-from-RNA + RNA-from-
protein, with the 0.015 switches), the optimisers and learning rates, gradient clipping, the x0.8 decay every 10 epochs and
the evidence-based early stopping (Model/Train_and_Predict.py::train, single-GPU behaviour), the preprocessing
(utils/Preprocessing.py: normalize_total 1e4, log1p, scale max 10 per sample; protein CLR per protein as muon's default)
and the graphs (utils/Graph_utils.py: spatial kNN k=6 united with an expression kNN k=10 - on PCA to 85% variance when
there are > 1500 genes - and self loops; the protein graph uses the protein kNN; edges i -> neighbours(i)).
Changes needed at our scale: graphs are built with a sparse GPU kNN instead of the dense N x N adjacency, per spatially
compact tile of <= TILE_TRAIN training cells (one tile = one graph = one optimisation step, like one Visium sample in the
original; prediction tiles <= TILE_PRED cells); no per-cell gene-count filter (min_genes=700 removes every cell of a
~400-gene Xenium panel); the gene filter of the original (detected in >= 2.5% of training cells, plus the targets' coding
genes) is used as in the original.
Predictions (CLR scale) are standardised with the training targets' mean and SD (kidney z-units for the cocktails).
modes:
  cross OUT_DIR GROUP[,GROUP...]   train on the kidney training stripes with the group's genes (panel-matched, like the
                                   other baselines); predict the group's cases -> OUT_DIR/bench_rows.csv, OUT_DIR/preds/
  within CASE [CASE ...]           train on the case's training stripes; predict; score the held-out stripes
                                   -> results/bench/within/CASE__dgat.csv
  xsec A B[,B...]                  cross-section tier: trained as in within mode on case A; predicts each case B of the same
                                   panel group (B's RNA scaled per sample, as any predicted sample) -> results/bench/xsec/
                                   (kidney sections: cross mode with KID_TRAIN=rcc|ccrcc)
"""
import os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, torch, torch.nn as nn, torch.optim as optim

from txcellnet.bench import (load_kidney, group_genes, group_cases, load_case_cells as load_case, subset_genes, score_cross,
                             within_split, write_within, xsec_train, xsec_test, write_xsec, PROTS, OWN, log)
sys.path.insert(0, os.environ.get('DGAT_DIR', 'external/DGAT'))
from Model.dgat import GATEncoder, Decoder_Protein, Decoder_mRNA

dev = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # TF32 tensor cores (speed only)
ALPHA, BETA, GAMMA, DELTA, ETA = 5.0, 1.0, 1.0, 3.0, 1.0  # Train_and_Predict.py
HID, DROP, EPOCHS, WARMUP, EB_THR = 1024, 0.3, 100, 50, 0.96
TILE_TRAIN, TILE_PRED = int(os.environ.get('TILE_TRAIN', 10000)), int(os.environ.get('TILE_PRED', 30000))
SEED = int(os.environ.get('SEED', 1)); MAX_GENES = None  # no cap: the original gene filter


def rna_scale(X):
    """Preprocessing.normalize on one sample: normalize_total(1e4), log1p, scale(max_value=10) -> dense float32."""
    X = X.astype(np.float32).tocsr(); tot = np.asarray(X.sum(1)).ravel()
    A = (sp.diags(np.where(tot > 0, 1e4 / np.maximum(tot, 1e-12), 0).astype(np.float32)) @ X).tocsr(); A.data = np.log1p(A.data)
    A = A.toarray(); mu = A.mean(0); sd = A.std(0, ddof=1); sd[sd == 0] = 1
    A -= mu; A /= sd; np.clip(A, -10, 10, out=A)
    return A


def clr0(P):
    """muon prot.pp.clr(axis=0): x -> log1p(x / exp(mean over cells of log1p x)), per protein."""
    P = np.asarray(P, np.float64); g = np.exp(np.log1p(P).mean(0, keepdims=True))
    return np.log1p(P / g).astype(np.float32)


def dgat_genes(X, genes, tr, own, max_genes=MAX_GENES):
    """qc_control_cytassist gene filter: detected in >= 2.5% of training cells, plus coding genes of the targets."""
    det = np.asarray((X[tr] > 0).mean(0)).ravel(); keep = det >= 0.025
    keep |= np.isin(np.array(genes), list(own))
    use = np.flatnonzero(keep)
    if max_genes and len(use) > max_genes:
        A = X[tr][:, use].astype(np.float32); A = (sp.diags(1e4 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A).tocsr(); A.data = np.log1p(A.data)
        m1 = np.asarray(A.mean(0)).ravel(); m2 = np.asarray(A.multiply(A).mean(0)).ravel(); use = np.sort(use[np.argsort(m2 - m1 ** 2)[::-1][:max_genes]])
    return [genes[i] for i in use]


def tiles(xy, size):
    """spatially compact tiles of ~size cells: x-strips, then y-chunks within each strip (indices into xy)."""
    n = len(xy); nt = max(1, int(round(n / size))); ns = max(1, int(round(np.sqrt(nt))))
    out = []
    for strip in np.array_split(np.argsort(xy[:, 0], kind='stable'), ns):
        oy = strip[np.argsort(xy[strip, 1], kind='stable')]
        out += [np.sort(t) for t in np.array_split(oy, max(1, int(round(len(strip) / size))))]
    return out


def knn_gpu(F, k, bs=2048):
    """k nearest neighbours (Euclidean, self included, as NearestNeighbors.kneighbors on the fitted data)."""
    F = torch.as_tensor(F, dtype=torch.float32, device=dev); k = min(k, len(F))
    return torch.cat([torch.cdist(F[i:i + bs], F).topk(k, largest=False).indices for i in range(0, len(F), bs)])


def pca85(X):
    """PCA(n_components=0.85, svd_solver='full') scores."""
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev); Xt = Xt - Xt.mean(0)
    U, S, _ = torch.linalg.svd(Xt, full_matrices=False); ev = S ** 2
    r = int(torch.searchsorted(torch.cumsum(ev, 0) / ev.sum(), torch.tensor([0.85], device=dev), right=True).item()) + 1
    return U[:, :r] * S[:r]


def edge_index(n, *nbrs):
    """nonzero() of (sum of kNN adjacencies + identity): edges i -> neighbour."""
    ar = torch.arange(n, device=dev)
    src = torch.cat([ar.repeat_interleave(nb.shape[1]) for nb in nbrs] + [ar]); dst = torch.cat([nb.reshape(-1) for nb in nbrs] + [ar])
    key = torch.unique(src * n + dst)
    return torch.stack([key // n, key % n])


def tile_graph(Xr, xy, Pp=None):
    n = len(Xr); snb = knn_gpu(xy, 6)
    feats = pca85(Xr) if Xr.shape[1] > 1500 else Xr
    g = dict(x=torch.as_tensor(Xr), ei=edge_index(n, snb, knn_gpu(feats, 10)).cpu())
    if Pp is not None: g['p'] = torch.as_tensor(Pp); g['pei'] = edge_index(n, snb, knn_gpu(Pp, 10)).cpu()
    return g


def rmse(a, b): return torch.sqrt(nn.functional.mse_loss(a, b) + 1e-8)


def train_dgat(graphs, n_genes, n_prot):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    enc_r = GATEncoder(in_channels=n_genes, hidden_dim=HID, dropout=DROP).to(dev); dec_r = Decoder_mRNA(HID, n_genes, dropout=0).to(dev)
    enc_p = GATEncoder(in_channels=n_prot, hidden_dim=HID, dropout=DROP).to(dev); dec_p = Decoder_Protein(HID, [f'p{i}' for i in range(n_prot)], dropout=0).to(dev)
    mods = [enc_r, dec_r, enc_p, dec_p]
    opts = [optim.Adam(enc_r.parameters(), lr=5e-4, weight_decay=2e-5), optim.Adam(dec_r.parameters(), lr=1e-4, weight_decay=1e-5),
            optim.Adam(enc_p.parameters(), lr=5e-4, weight_decay=2e-5), optim.Adam(dec_p.parameters(), lr=1e-4, weight_decay=1e-5)]
    D = sum(p.numel() for m in mods for p in m.parameters()); evid = []; t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        for m in mods: m.train()
        tot = 0.0
        for gi in rng.permutation(len(graphs)):
            g = {k: v.to(dev, non_blocking=True) for k, v in graphs[gi].items()}
            for o in opts: o.zero_grad()
            z_r = enc_r(g['x'], g['ei']); z_p = enc_p(g['p'], g['pei'])
            l_rr = rmse(dec_r(z_r), g['x']); l_pp = rmse(dec_p(z_p), g['p'])
            l_al = nn.functional.mse_loss(z_r, z_p)  # single device: z_protein.to(device) under no_grad returns z_protein itself
            l_pr = rmse(dec_p(z_r), g['p']); l_rp = rmse(dec_r(z_p), g['x'])
            loss = (ALPHA * l_rr + (0.0 if l_pp < 0.015 else BETA) * l_pp + (0.0 if l_al < 0.015 else GAMMA) * l_al
                    + (0.0 if l_pr < 0.015 else DELTA) * l_pr + ETA * l_rp)
            loss.backward()
            for m in mods: torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            for o in opts: o.step()
            tot += loss.item()
        if ep % 10 == 0:
            for o in opts:
                for pg in o.param_groups: pg['lr'] *= 0.8
        s = 0.0  # evidence-based criterion on the last step's gradients, as in the original
        for o in opts:
            for pg in o.param_groups:
                for p in pg['params']:
                    if p.grad is None: continue
                    st = o.state[p]; m_t, v_t = st.get('exp_avg'), st.get('exp_avg_sq')
                    if m_t is None or v_t is None: continue
                    s += (p.grad.pow(2) / (v_t - m_t.pow(2) + 1e-8)).sum().item()
        evid = (evid + [1.0 - s / D])[-10:]
        if ep % 5 == 0 or ep == 1:
            log(f'epoch {ep} loss {tot / len(graphs):.4f} (last: rna {l_rr.item():.3f} prot {l_pp.item():.3f} align {l_al.item():.3f} '
                f'prot<-rna {l_pr.item():.3f}) evidence {evid[-1]:.4f} {time.time() - t0:.0f}s')
        if ep > WARMUP and np.mean(evid) > EB_THR:
            log(f'EB early stop at epoch {ep}'); break
    return enc_r, dec_p


@torch.no_grad()
def predict(enc_r, dec_p, Xr, xy, n_prot):
    enc_r.eval(); dec_p.eval(); out = np.zeros((len(Xr), n_prot), np.float32)
    for t in tiles(xy, TILE_PRED):
        g = tile_graph(Xr[t], xy[t])
        out[t] = dec_p(enc_r(g['x'].to(dev), g['ei'].to(dev))).cpu().numpy()
    return out


def run_cross(OUT, groups):
    os.makedirs(f'{OUT}/preds', exist_ok=True); kid = load_kidney(); rows = []
    rp = f'{OUT}/bench_rows.csv'
    if os.path.exists(rp): rows = pd.read_csv(rp).to_dict('records')
    own = {g for p in PROTS for g in OWN.get(p, [])}
    for gname in groups:
        genes0 = group_genes(gname, kid)
        Gtr = sp.vstack([subset_genes(k['G'], k['genes'], genes0)[k['tr']] for k in kid]).tocsr()
        genes = dgat_genes(Gtr, genes0, np.arange(Gtr.shape[0]), own)
        graphs, Ptr = [], []
        for k in kid:
            if not k['tr'].any(): continue  # KID_TRAIN: the other section has no training cells
            Xr = rna_scale(subset_genes(k['G'], k['genes'], genes)); tr = np.flatnonzero(k['tr'])
            Pc = clr0(k['P'][tr]); Ptr.append(Pc)
            for t in tiles(k['xy'][tr], TILE_TRAIN): graphs.append(tile_graph(Xr[tr[t]], k['xy'][tr[t]], Pc[t]))
            del Xr
        Ptr = np.vstack(Ptr); mu, sd = Ptr.mean(0), Ptr.std(0) + 1e-6
        log(gname, 'genes', len(genes), 'of', len(genes0), 'training graphs', len(graphs), 'cells', sum(len(g['x']) for g in graphs))
        enc_r, dec_p = train_dgat(graphs, len(genes), len(PROTS)); del graphs
        for c in group_cases(gname):
            case = load_case(c); cells = case['cells']
            node = np.flatnonzero((cells['area_px'].values >= 10) & (np.asarray(case['X'].sum(1)).ravel() >= 1))
            Xq = rna_scale(subset_genes(case['X'], case['genes'], genes)[node])
            pz = np.full((len(cells), len(PROTS)), np.nan, np.float32)
            pz[node] = (predict(enc_r, dec_p, Xq, case['xy'][node], len(PROTS)) - mu) / sd
            np.save(f'{OUT}/preds/{c}__dgat_pm.npy', pz.astype(np.float16))
            rows = [r for r in rows if not (r['case'] == case['meta']['dataset'] and r['method'] == 'dgat_pm')]
            rows += score_cross(case, pz, 'dgat_pm', gname, len(genes))
            df = pd.DataFrame(rows); df.to_csv(rp, index=False)
            log(c, 'dgat_pm median Spearman', round(float(df[df.case == case['meta']['dataset']].spearman.median()), 3))
        del enc_r, dec_p; torch.cuda.empty_cache()


def run_within(cases):
    for c in cases:
        case = load_case(c); tr, te, keys, T, R = within_split(case); cells = case['cells']
        own = {g for k in keys for g in case['meta']['markers'][k].get('own', [])}
        genes = dgat_genes(case['X'], case['genes'], tr, own)
        node = np.flatnonzero((cells['area_px'].values >= 10) & (np.asarray(case['X'].sum(1)).ravel() >= 1))
        pos = np.full(len(cells), -1); pos[node] = np.arange(len(node))
        Xr = rna_scale(subset_genes(case['X'], case['genes'], genes)[node]); xy = case['xy'][node]
        Pc = clr0(R[tr]); mu, sd = Pc.mean(0), Pc.std(0) + 1e-6
        trn = pos[tr]; graphs = [tile_graph(Xr[trn[t]], xy[trn[t]], Pc[t]) for t in tiles(xy[trn], TILE_TRAIN)]
        log(c, 'train', len(tr), 'test', len(te), 'genes', len(genes), 'targets', keys, 'graphs', len(graphs))
        enc_r, dec_p = train_dgat(graphs, len(genes), len(keys)); del graphs
        pred = np.full((len(cells), len(keys)), np.nan, np.float32)
        pred[node] = (predict(enc_r, dec_p, Xr, xy, len(keys)) - mu) / sd
        write_within(c, 'dgat', keys, pred, T, te)
        del enc_r, dec_p; torch.cuda.empty_cache()


def run_xsec(a, bs):
    A, trA, kA, TA, RA = xsec_train(a); cells = A['cells']
    own = {g for k in kA for g in A['meta']['markers'][k].get('own', [])}
    genes = dgat_genes(A['X'], A['genes'], trA, own)
    node = np.flatnonzero((cells['area_px'].values >= 10) & (np.asarray(A['X'].sum(1)).ravel() >= 1))
    pos = np.full(len(cells), -1); pos[node] = np.arange(len(node))
    Xr = rna_scale(subset_genes(A['X'], A['genes'], genes)[node]); xy = A['xy'][node]
    Pc = clr0(RA[trA]); mu, sd = Pc.mean(0), Pc.std(0) + 1e-6
    trn = pos[trA]; graphs = [tile_graph(Xr[trn[t]], xy[trn[t]], Pc[t]) for t in tiles(xy[trn], TILE_TRAIN)]
    log(a, 'xsec train', len(trA), 'genes', len(genes), 'targets', kA, 'graphs', len(graphs))
    enc_r, dec_p = train_dgat(graphs, len(genes), len(kA)); del graphs, Xr
    for b in bs:
        B, keys, ia, teB, okB, TB = xsec_test(b, kA); cb = B['cells']
        nodeB = np.flatnonzero((cb['area_px'].values >= 10) & (np.asarray(B['X'].sum(1)).ravel() >= 1))
        XrB = rna_scale(subset_genes(B['X'], B['genes'], genes)[nodeB])
        pred = np.full((len(cb), len(kA)), np.nan, np.float32)
        pred[nodeB] = (predict(enc_r, dec_p, XrB, B['xy'][nodeB], len(kA)) - mu) / sd; del XrB
        write_xsec(a, b, 'dgat', keys, pred[:, ia], TB, teB, okB)
    del enc_r, dec_p; torch.cuda.empty_cache()


if __name__ == '__main__':
    if sys.argv[1] == 'cross': run_cross(sys.argv[2], sys.argv[3].split(','))
    elif sys.argv[1] == 'within': run_within(sys.argv[2:])
    elif sys.argv[1] == 'xsec': run_xsec(sys.argv[2], sys.argv[3].split(','))
    else: raise SystemExit(__doc__)
