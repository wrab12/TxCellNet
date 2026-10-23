"""SpaMosaic (Yan et al., Nat Genet 2026; github.com/JinmiaoChenLab/SpaMosaic, commit cc13367) on our benchmark, following
the official RNA + ADT demo (demo/Human_lymph.ipynb): RNA_preprocess(batch_corr=True, favor='scanpy', n_hvg=5000),
ADT_preprocess(batch_corr=True) (per-cell CLR, PCA, Harmony), SpaMosaic(intra_knns=10, inter_knn_base=10, w_g=0.8,
seed=1234), train(net='wlgcn', lr=0.01, T=0.01, n_epochs=100), impute(imp_knn=10). Samples with RNA and protein are
bridges, the query sample has RNA only; a query cell's protein is the mean over its 10 nearest reference cells in the
aligned space of the layer 'counts', which we set to our standard targets (log1p, z-scored per reference sample), so the
predictions are on the scale of the other baselines. The query RNA is used for training (transductive).
If training diverges (non-finite contrastive loss), it is restarted with the next seed (1235, 1236).
Genes: all genes of the panel (cross: the panel-matched genes); RNA_preprocess selects its 5,000 highly variable genes as
in the demo.
Changes: the spatial graph builder is vectorised (same graph as build_graph.Cal_Spatial_Net); reference samples are
subsampled to <= N_REF training cells each (the contrastive loss keeps a 2,048 x 2,048 similarity matrix per 1,024-cell
mini-batch for the backward pass); query cells are not subsampled.
modes (as dgat_bench.py; one model per query case):
  cross OUT_DIR GROUP[,GROUP...]   bridges = kidney training stripes of both sections -> OUT_DIR/bench_rows.csv, preds/
  within CASE [CASE ...]           bridge = training stripes, query = held-out stripes -> results/bench/within/CASE__spamosaic.csv
  xsec A B[,B...]                  cross-section tier: bridge = training stripes of case A (as in its within run), query = every
                                   evaluated cell of case B (same panel group); one model per B -> results/bench/xsec/
                                   (kidney sections: cross mode with KID_TRAIN=rcc|ccrcc)
"""
import os, sys, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # as in the demo (deterministic cuBLAS)
import numpy as np, pandas as pd, scipy.sparse as sp, scanpy as sc, torch
from sklearn.neighbors import NearestNeighbors

from txcellnet.bench import (load_kidney, group_genes, group_cases, load_case_cells as load_case, subset_genes, score_cross,
                             within_split, write_within, xsec_train, xsec_test, write_xsec, PROTS, log)
sys.path.insert(0, os.environ.get('SPAMOSAIC_DIR', 'external/SpaMosaic'))
import spamosaic.build_graph as build_graph
from spamosaic.framework import SpaMosaic
from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess

N_REF = int(os.environ.get('N_REF', 100_000))


def cal_spatial_net(adata, rad_cutoff=None, k_cutoff=None, max_neigh=50, model='Radius', verbose=True):
    """Vectorised build_graph.Cal_Spatial_Net: the max_neigh nearest neighbours (excluding the first, i.e. self) within
    rad_cutoff ('Radius') or the k_cutoff nearest ('KNN'), plus self loops -> adata.uns['adj']."""
    xy = np.asarray(adata.obsm['spatial'], np.float64)
    d, ind = NearestNeighbors(n_neighbors=min(max_neigh + 1, len(xy))).fit(xy).kneighbors(xy)
    d, ind = (d[:, 1:k_cutoff + 1], ind[:, 1:k_cutoff + 1]) if model == 'KNN' else (d[:, 1:], ind[:, 1:])
    keep = (d < rad_cutoff).ravel() if model == 'Radius' else np.ones(d.size, bool)
    r = np.repeat(np.arange(len(xy)), ind.shape[1])[keep]; c = ind.ravel()[keep]
    adata.uns['adj'] = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(len(xy), len(xy))) + sp.eye(len(xy))
    if verbose: print(f'spatial graph: {len(r)} edges, {len(xy)} cells')


build_graph.Cal_Spatial_Net = cal_spatial_net
for _cls in (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix):  # .A was removed in SciPy 1.14; SpaMosaic pins SciPy 1.10
    if not hasattr(_cls, 'A'): _cls.A = property(lambda self: self.toarray())


def sub(idx, n, seed=0):
    return idx if len(idx) <= n else np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))


def impute(refs, qname, Xq, xyq, genes, names):
    """refs: [(name, counts csr, xy, raw protein (for the ADT embedding), targets in z-units)]; query: counts Xq, xy.
    Returns the imputed targets of the query cells."""
    rna, adt = [], []
    for name, X, xy, Praw, Y in refs:
        idx = [f'{name}_{i}' for i in range(X.shape[0])]
        a = sc.AnnData(X.astype(np.float32).tocsr(), obs=pd.DataFrame({'src': name}, index=idx)); a.var_names = genes; a.obsm['spatial'] = xy
        a.layers['counts'] = a.X  # impute() stacks layer 'counts' of every modality
        b = sc.AnnData(np.asarray(Praw, np.float32), obs=pd.DataFrame({'src': name}, index=idx)); b.var_names = names; b.obsm['spatial'] = xy
        b.layers['counts'] = np.asarray(Y, np.float32); rna.append(a); adt.append(b)
    q = sc.AnnData(Xq.astype(np.float32).tocsr(), obs=pd.DataFrame({'src': qname}, index=[f'{qname}_{i}' for i in range(Xq.shape[0])]))
    q.var_names = genes; q.obsm['spatial'] = xyq; q.layers['counts'] = q.X; rna.append(q); adt.append(None)
    t0 = time.time()
    RNA_preprocess(rna, batch_corr=True, favor='scanpy', n_hvg=5000, batch_key='src', key='dimred_bc')
    ADT_preprocess(adt, batch_corr=True, batch_key='src', key='dimred_bc')
    D = {'rna': rna, 'adt': adt}
    for seed in (1234, 1235, 1236):  # official default seed; restart with the next seed if training diverges
        model = SpaMosaic(modBatch_dict=D, input_key='dimred_bc', batch_key='src', intra_knns=10, inter_knn_base=10, w_g=0.8,
                          seed=seed, device='cuda:0')
        model.train(net='wlgcn', lr=0.01, T=0.01, n_epochs=100)
        if np.isfinite(model.loss_cl).all() and np.isfinite(model.loss_rec).all(): break
        log('SpaMosaic training diverged with seed', seed); torch.cuda.empty_cache()
    else:  # no seed converged: no result (as for lymph node 5K and glioblastoma within datasets)
        torch.use_deterministic_algorithms(False)
        raise FloatingPointError('SpaMosaic training diverged with seeds 1234-1236')
    model.infer_emb(D, emb_key='emb', final_latent_key='merged_emb')
    out = model.impute(D, emb_key='emb', layer_key='counts', imp_knn=10)['adt'][len(refs)]
    log('SpaMosaic', 'bridges', [r[1].shape[0] for r in refs], 'query', Xq.shape[0], 'genes', len(genes),
        'loss_cl', round(model.loss_cl[-1], 3) if model.loss_cl else None, f'{time.time() - t0:.0f}s')
    torch.use_deterministic_algorithms(False)
    return np.asarray(out, np.float32)


def run_cross(OUT, groups):
    os.makedirs(f'{OUT}/preds', exist_ok=True); kid = load_kidney(); rp = f'{OUT}/bench_rows.csv'
    rows = pd.read_csv(rp).to_dict('records') if os.path.exists(rp) else []
    for gname in groups:
        genes = group_genes(gname, kid); refs = []
        for k in kid:
            if not k['tr'].any(): continue  # KID_TRAIN: the other section has no training cells
            t = sub(np.flatnonzero(k['tr']), N_REF); L = np.log1p(k['P'][t].astype(np.float64))
            refs.append((k['name'], subset_genes(k['G'], k['genes'], genes)[t], k['xy'][t], k['P'][t], (L - L.mean(0)) / (L.std(0) + 1e-6)))
        for c in group_cases(gname):
            case = load_case(c); cells = case['cells']
            # query = cells with RNA (scored cells for the kidney sections, whose other cells are reference)
            node = np.flatnonzero(case['ok']) if c.startswith('kidney') else \
                np.flatnonzero((cells['area_px'].values >= 10) & (np.asarray(case['X'].sum(1)).ravel() >= 1))
            Xq = subset_genes(case['X'], case['genes'], genes)[node]
            pz = np.full((len(cells), len(PROTS)), np.nan, np.float32)
            pz[node] = impute(refs, 'query', Xq, case['xy'][node], genes, PROTS)
            np.save(f'{OUT}/preds/{c}__spamosaic_pm.npy', pz.astype(np.float16))
            rows = [r for r in rows if not (r['case'] == case['meta']['dataset'] and r['method'] == 'spamosaic_pm')]
            rows += score_cross(case, pz, 'spamosaic_pm', gname, len(genes))
            df = pd.DataFrame(rows); df.to_csv(rp, index=False)
            log(c, gname, 'spamosaic_pm median Spearman', round(float(df[df.case == case['meta']['dataset']].spearman.median()), 3))


def run_within(cases):
    for c in cases:
        case = load_case(c); tr, te, keys, T, R = within_split(case)
        genes = list(case['genes']); X = case['X']  # all panel genes: RNA_preprocess selects its 5,000 HVGs (demo)
        mu, sd = T[tr].mean(0), T[tr].std(0) + 1e-6; trs = sub(tr, N_REF)
        log(c, 'train', len(tr), '(used', len(trs), ') test', len(te), 'genes', len(genes), 'targets', keys)
        names = [f'p{j}' for j in range(len(keys))]
        pred = np.full((len(case['cells']), len(keys)), np.nan, np.float32)
        pred[te] = impute([('train', X[trs], case['xy'][trs], R[trs], (T[trs] - mu) / sd)], 'test', X[te], case['xy'][te], genes, names)
        write_within(c, 'spamosaic', keys, pred, T, te)


def run_xsec(a, bs):
    A, trA, kA, TA, RA = xsec_train(a); genes = list(A['genes'])  # all panel genes; RNA_preprocess selects 5,000 HVGs
    mu, sd = TA[trA].mean(0), TA[trA].std(0) + 1e-6; trs = sub(trA, N_REF); names = [f'p{j}' for j in range(len(kA))]
    for b in bs:
        B, keys, ia, teB, okB, TB = xsec_test(b, kA)
        log(a, '->', b, 'bridge', len(trs), 'query', len(okB), 'genes', len(genes), 'targets', keys)
        pred = np.full((len(B['cells']), len(kA)), np.nan, np.float32)
        try:
            pred[okB] = impute([('bridge', A['X'][trs], A['xy'][trs], RA[trs], (TA[trs] - mu) / sd)], 'query',
                               subset_genes(B['X'], B['genes'], genes)[okB], B['xy'][okB], genes, names)
        except FloatingPointError as e:  # recorded as no result; the next pair still runs
            log(a, '->', b, 'no result:', e); open(f'results/bench/xsec/{a}__{b}__spamosaic.diverged', 'w').write(str(e) + '\n')
            torch.cuda.empty_cache(); continue
        write_xsec(a, b, 'spamosaic', keys, pred[:, ia], TB, teB, okB)


if __name__ == '__main__':
    if sys.argv[1] == 'cross': run_cross(sys.argv[2], sys.argv[3].split(','))
    elif sys.argv[1] == 'within': run_within(sys.argv[2:])
    elif sys.argv[1] == 'xsec': run_xsec(sys.argv[2], sys.argv[3].split(','))
    else: raise SystemExit(__doc__)
