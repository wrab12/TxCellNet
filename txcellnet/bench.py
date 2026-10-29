"""Benchmark data access shared by the scripts, so that every method sees exactly the same cells, splits, targets and
metrics. All paths are relative to the repository root (data/ and results/); files are read on first use.

Kidney metadata (training grids written by scripts/data/prep_kidney_grid.py):
  PROTS  the 27 kidney proteins (output channels of the kidney networks), OWN  protein -> coding genes,
  VOCAB  the training vocabulary (union of the two kidney panels)
Benchmark tables: PANELS  panel groups (results/bench/panels.json, scripts/data/make_panels.py),
  CW  stain-cocktail weights (results/bench/cocktail_weights.json, scripts/data/fit_cocktail.py)

Cross-dataset setting: training cells = kidney training stripes (both sections), genes = the panel group's genes measured in
both sections, targets = 10x per-cell protein values; predictions are scored per marker, stains through combine().
Same-section setting: training cells = even x-stripes of a case (split.json), test cells = odd stripes, targets = the
case's measured per-cell values (meta.json markers that have an image target in Y.json).
"""
import functools
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from .metrics import metrics
from .utils import log

KIDNEY = {'rcc': 'data/xenium_protein_rcc/pilot_L2', 'ccrcc': 'data/xenium_protein_ccrcc/pilot_L2'}
XSEC = 'results/bench/xsec'

# ---------------- section lists (Supplementary Table S11) ----------------
BREAST = lambda pts: [f'xen_breast_S{p}_{r}' for p in pts for r in ('Top', 'Mid', 'Bot')]
KID = ['kidney_rcc', 'kidney_ccrcc']
DEV = BREAST((1,)) + ['xen_lung_v1', 'xen_pancreas_v1', 'crlm_tumor1', 'crlm_tumor3_A1', 'duchini_core01_hcc']
TEST = BREAST((2, 3, 4)) + ['xen_lymphnode_5k', 'crlm_tumor2', 'crlm_tumor3_A2', 'crlm_tumor3_A3', 'crlm_adjnormal',
                            'duchini_core02_non_tumour', 'duchini_core03_tonsil', 'duchini_core04_hca']
FRESH = ['xen_skin_5k', 'xen_prostate_5k', 'xen_ovary_5k', 'xen_ovary_v1ff', 'xen_breast_5k', 'xen_cervix_5k', 'xen_gbm_v1']

# ---------------- spot-level evaluation (scripts/baselines/grid_eval.py, scripts/evaluation/grid_summary.py) ----------------
GRID_GROUPS_WITHIN = [('Kidney (IF)', ['kidney_rcc', 'kidney_ccrcc']),
                      ('CRLM (IMC)', ['crlm_tumor1', 'crlm_tumor2', 'crlm_adjnormal', 'crlm_tumor3_A1', 'crlm_tumor3_A2', 'crlm_tumor3_A3']),
                      ('Liver, tonsil (COMET)', ['duchini_core01_hcc', 'duchini_core02_non_tumour', 'duchini_core03_tonsil', 'duchini_core04_hca']),
                      ('Breast (stains)', BREAST((1, 2, 3, 4))), ('Lung, pancreas (stains)', ['xen_lung_v1', 'xen_pancreas_v1']),
                      ('Lymph node, fresh (stains)', ['xen_lymphnode_5k'] + FRESH)]
GRID_GROUPS_CROSS = [('Breast (stains)', BREAST((2, 3, 4))), ('Lymph node (stains)', ['xen_lymphnode_5k']),
                     ('CRLM (IMC)', ['crlm_tumor2', 'crlm_tumor3_A2', 'crlm_tumor3_A3', 'crlm_adjnormal']),
                     ('Liver, tonsil (COMET)', ['duchini_core02_non_tumour', 'duchini_core03_tonsil', 'duchini_core04_hca']),
                     ('Fresh (stains)', FRESH)]
GRID_METHODS = [('txcellnet', 'TxCellNet'), ('nichetrans', 'NicheTrans'), ('dgat_cell', 'DGAT, per cell'), ('dgat_spot', 'DGAT, on spots')]


# ---------------- lazily loaded tables ----------------
@functools.lru_cache(maxsize=None)
def kidney_meta(section='rcc'):
    return json.load(open(f'{KIDNEY[section]}/meta.json'))


def prots():
    return kidney_meta('rcc')['prots']


def own_genes():
    return kidney_meta('rcc')['own_genes']


def vocab():
    return sorted(set(kidney_meta('rcc')['genes']) | set(kidney_meta('ccrcc')['genes']))


@functools.lru_cache(maxsize=None)
def panels():
    return json.load(open('results/bench/panels.json'))


@functools.lru_cache(maxsize=None)
def cocktail_weights():
    return json.load(open('results/bench/cocktail_weights.json'))['mean']


def __getattr__(name):  # module attributes PROTS, OWN, VOCAB, PANELS, CW, read on first access
    lazy = {'PROTS': prots, 'OWN': own_genes, 'VOCAB': vocab, 'PANELS': panels, 'CW': cocktail_weights}
    if name in lazy: return lazy[name]()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def combine(pc, comps, key):
    """prediction of marker `key` from per-cell predictions pc (cells x PROTS): a single protein, or the kidney-fitted
    weighted sum of the component proteins for a stain cocktail"""
    P = prots()
    if len(comps) == 1: return pc[:, P.index(comps[0])]
    w = cocktail_weights().get(key, {}).get('weights', [1.0] * len(comps))
    return sum(wi * pc[:, P.index(c)] for wi, c in zip(w, comps))


def bench_rows(pattern='results/bench/*/bench_rows.csv'):
    """all benchmark rows, one per (case, marker, method); directories are read in name order and a later directory
    overrides an earlier one for the same row (a repeated row is reported)"""
    import glob
    fs = sorted(glob.glob(pattern))
    R = pd.concat([pd.read_csv(f) for f in fs])
    dup = R.duplicated(['case', 'marker', 'method'], keep='last')
    if dup.any(): log('bench_rows:', int(dup.sum()), 'repeated rows, the last directory in name order is kept')
    return R[~dup]


# ---------------- cross-dataset setting ----------------
def load_kidney():
    """Both kidney sections: raw gene counts (csr), gene names, raw protein counts (cells x PROTS, int), centroids (um),
    training-stripe mask (cells with >= 1 count), cell ids.
    Env KID_TRAIN=rcc|ccrcc (other section of the same dataset): only that section's training stripes are training cells;
    the other section is still loaded (genes measured in both sections) but contributes no training cells."""
    import h5py
    out = []; only = os.environ.get('KID_TRAIN', ''); P_ = prots()
    for s in ['rcc', 'ccrcc']:
        with h5py.File(f'data/xenium_protein_{s}/outs/cell_feature_matrix.h5', 'r') as f:
            m = f['matrix']
            bc = np.array([b.decode() for b in m['barcodes'][:]]); ft = np.array([t.decode() for t in m['features']['feature_type'][:]])
            nm = np.array([t.decode() for t in m['features']['name'][:]])
            X = sp.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=tuple(m['shape'][:])).T.tocsr()
        G = X[:, ft == 'Gene Expression'].tocsr(); gn = list(nm[ft == 'Gene Expression'])
        P = X[:, ft == 'Protein Expression'].toarray(); pn = list(nm[ft == 'Protein Expression'])
        P = P[:, [pn.index(p) for p in P_]]
        cells = pd.read_parquet(f'data/xenium_protein_{s}/outs/cells.parquet').set_index('cell_id').reindex(bc)
        xy = cells[['x_centroid', 'y_centroid']].values.astype(np.float64)
        edges = np.array(json.load(open(f'results/cell_baseline/{s}_stripes.json'))['x_edges_um'])
        tr = (np.digitize(xy[:, 0], edges) % 2 == 0) & (np.asarray(G.sum(1)).ravel() > 0)
        if only and s != only: tr[:] = False
        out.append(dict(name=f'kidney_{s}', G=G, genes=gn, P=P, xy=xy, tr=tr, ids=bc))
        log('kidney', s, 'cells', len(bc), 'train', int(tr.sum()))
    return out


def group_genes(gname, kid):
    both = set(kid[0]['genes']) & set(kid[1]['genes'])
    return [g for g in panels()[gname]['genes'] if g in both]


def group_cases(gname):
    """the group's prepared cases (Visium HD excluded); env CASES=a,b restricts them (one job per case)."""
    only = [c for c in os.environ.get('CASES', '').split(',') if c]
    return [c for c in panels()[gname]['cases'] if not c.startswith('vhd') and os.path.exists(f'data/cases/{c}/meta.json')
            and (not only or c in only)]


def load_case_cells(case):
    """cells.parquet, csr counts (rows = cells.parquet), gene names, meta, evaluated-cell mask, centroids (um)."""
    cd = f'data/cases/{case}'; meta = json.load(open(f'{cd}/meta.json')); cells = pd.read_parquet(f'{cd}/cells.parquet')
    Xc = sp.load_npz(f'{cd}/counts.npz').tocsr(); cg = json.load(open(f'{cd}/counts_genes.json'))
    ok = (cells['area_px'].values >= 10) & (np.asarray(Xc.sum(1)).ravel() >= 1)
    if 'eval_mask' in meta: ok &= cells[meta['eval_mask']].values.astype(bool)
    return dict(name=case, cells=cells, X=Xc, genes=cg, meta=meta, ok=ok, xy=cells[['x_um', 'y_um']].values.astype(np.float64))


def subset_genes(X, genes, use):
    """csr counts restricted / reordered to `use` (genes absent from `genes` -> zero columns)."""
    pos = {g: i for i, g in enumerate(genes)}
    have = [j for j, g in enumerate(use) if g in pos]
    A = X[:, [pos[use[j]] for j in have]].tocsr()
    if len(have) == len(use): return A
    M = sp.lil_matrix((len(use), len(have)), dtype=np.float32)
    for k, j in enumerate(have): M[j, k] = 1
    return (A @ M.T.tocsr()).tocsr()


def score_cross(case, preds, method, gname, n_genes):
    """preds: cells x PROTS (rows = cells.parquet; NaN allowed outside the evaluated cells) in kidney z-units."""
    meta, cells = case['meta'], case['cells']; ok = case['ok'] & np.isfinite(preds).all(1)
    rows = []
    for key, mk in meta['markers'].items():
        tt = cells[mk['truth']].values.astype(np.float64)
        if mk.get('log', True): tt = np.log1p(np.clip(tt - mk.get('offset', 0), 0, None))
        rows.append(dict(case=meta['dataset'], marker=key, method=method, panel=gname, n_genes=n_genes,
                         **metrics(combine(preds, mk['components'], key)[ok], tt[ok]), n=int(ok.sum())))
    return rows


# ---------------- same-section setting ----------------
def within_split(case):
    """Split of the same-section setting: training / test cell indices (rows of cells.parquet), marker keys, targets T
    (log1p, as scored) and the raw non-negative intensities R (clip(t - offset, 0)) the targets derive from."""
    cd = f'data/cases/{case["name"]}'; yj = json.load(open(f'{cd}/Y.json')); meta, cells = case['meta'], case['cells']
    lab = np.load(f'{cd}/lab.npy', mmap_mode='r')
    n = int(max(np.asarray(lab[r0:r0 + 1024]).max() for r0 in range(0, lab.shape[0], 1024))) + 1
    xs_sum = np.zeros(n); cnt = np.zeros(n)
    for r0 in range(0, lab.shape[0], 1024):  # per-cell mean x of the label image, in blocks (lab can be ~20k x 13k)
        blk = np.asarray(lab[r0:r0 + 1024]); yy, xx = np.nonzero(blk); lv = blk[yy, xx]
        xs_sum += np.bincount(lv, weights=xx, minlength=n); cnt += np.bincount(lv, minlength=n)
    xmean = xs_sum / np.maximum(cnt, 1)
    edges = np.array(json.load(open(f'{cd}/split.json'))['x_edges_px'])
    test = np.digitize(xmean[cells['label'].values], edges) % 2 == 1
    tr, te = np.flatnonzero(case['ok'] & ~test), np.flatnonzero(case['ok'] & test)
    keys = [k for k in meta['markers'] if k in yj['markers']]
    T, R = [], []
    for k in keys:
        mk = meta['markers'][k]; t = cells[mk['truth']].values.astype(np.float64)
        r = np.clip(t - mk.get('offset', 0), 0, None) if mk.get('log', True) else np.expm1(np.clip(t, 0, None))
        R.append(r); T.append(np.log1p(r) if mk.get('log', True) else t)
    return tr, te, keys, np.stack(T, 1), np.stack(R, 1)


def within_genes(case, tr, max_genes=2000):
    """Genes of within_baselines.py: detected in >= 0.1% of training cells, at most the 2,000 most variable
    (log1p(normalize_total 1e3) on the training cells)."""
    X = case['X']; det = np.asarray((X[tr] > 0).mean(0)).ravel(); use = np.flatnonzero(det >= 1e-3)
    if len(use) > max_genes:
        A = X[tr][:, use].astype(np.float32); A = (sp.diags(1e3 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A).tocsr(); A.data = np.log1p(A.data)
        m1 = np.asarray(A.mean(0)).ravel(); m2 = np.asarray(A.multiply(A).mean(0)).ravel(); use = np.sort(use[np.argsort(m2 - m1 ** 2)[::-1][:max_genes]])
    return [case['genes'][i] for i in use]


def write_within(case_name, method, keys, pred, T, te, extra=None):
    """pred: cells x keys (any monotone scale); Spearman on the test cells -> results/bench/within/CASE__METHOD.csv."""
    os.makedirs('results/bench/within', exist_ok=True)
    rows = [dict(case=case_name, marker=k, method=f'within:{method}', spearman=float(spearmanr(pred[te, j], T[te, j])[0]), n=len(te), **(extra or {}))
            for j, k in enumerate(keys)]
    df = pd.DataFrame(rows); df.to_csv(f'results/bench/within/{case_name}__{method}.csv', index=False)
    os.makedirs(f'data/cases/{case_name}/runs/within_baselines', exist_ok=True)
    np.save(f'data/cases/{case_name}/runs/within_baselines/{method}.npy', pred.astype(np.float32))
    log(case_name, method, 'median Spearman', round(float(df.spearman.median()), 3), dict(zip(keys, df.spearman.round(3))))
    return df


# ---------------- other section of the same dataset ----------------
def xsec_train(a):
    """Training side: the same-section training stripes of case a (same cells and targets as its same-section run).
    Returns the load_case_cells dict, training cell indices, marker keys, targets T (log1p) and raw intensities R."""
    A = load_case_cells(a); trA, _, kA, TA, RA = within_split(A)
    return A, trA, kA, TA, RA


def xsec_test(b, kA):
    """Test side: case b of the same panel group. Returns the load_case_cells dict, the keys of a also measured in b, their
    column indices in a's keys, b's held-out stripes (the test cells of its same-section run), all its evaluated cells, and
    b's targets for those keys."""
    B = load_case_cells(b); _, teB, kB, TB, _ = within_split(B)
    keys = [k for k in kA if k in kB]
    return B, keys, [kA.index(k) for k in keys], teB, np.flatnonzero(B['ok']), TB[:, [kB.index(k) for k in keys]]


def write_xsec(a, b, method, keys, pred, TB, teB, okB):
    """pred: cells of b (rows = cells.parquet) x keys, NaN where not predicted. Scores b's held-out stripes ('heldout', the
    cells of its same-section run) and all its evaluated cells ('all') -> results/bench/xsec/A__B__METHOD.csv (Spearman,
    Pearson, AUROC); predictions -> results/bench/xsec/preds/A__B__METHOD.npy."""
    os.makedirs(f'{XSEC}/preds', exist_ok=True)
    rows = [dict(train=a, test=b, marker=k, method=method, cells=name, **metrics(pred[idx, j], TB[idx, j]), n=len(idx))
            for name, idx in (('heldout', teB), ('all', okB)) for j, k in enumerate(keys)]
    df = pd.DataFrame(rows); df.to_csv(f'{XSEC}/{a}__{b}__{method}.csv', index=False)
    np.save(f'{XSEC}/preds/{a}__{b}__{method}.npy', pred.astype(np.float16))
    med = df.groupby('cells').spearman.median().round(3).to_dict()
    log(a, '->', b, method, 'median Spearman: held-out stripes', med.get('heldout'), '| all cells', med.get('all'))
    return df
