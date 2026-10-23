"""Cognate-mRNA panels and metrics for the all-method image figure (Supplementary Fig. S3).
CRLM window of crlm_tumor3_A3_allm.npz (allmethods_extract.py): its offset in the case's label image is recovered from the
labels, and the cognate density is computed from the molecules of the protein's coding genes (Gaussian sigma = 2 px).
Per-marker cell-level metrics of the cognate mRNA ('own_mRNA', training-free) for kidney_rcc and crlm_tumor3_A3, checked against the
cell count of the trained methods.
usage: allmethods_cognate.py OUT_DIR    (OUT_DIR: the FIGDATA directory written by allmethods_extract.py)
"""
import json, sys
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter

from txcellnet.bench import OWN

OUT = sys.argv[1]; case = 'crlm_tumor3_A3'; cd = f'data/cases/{case}'
C = np.load(f'{OUT}/crlm_tumor3_A3_allm.npz'); lw = C['lab']; marks = [str(m) for m in C['marks']]; Zs = lw.shape[0]
lab = np.load(f'{cd}/lab.npy')
border = set(np.unique(np.concatenate([lw[0], lw[-1], lw[:, 0], lw[:, -1]])).tolist())
cand = [l for l in np.unique(lw) if l > 0 and l not in border]; L = cand[len(cand) // 2]  # a cell fully inside the window
wy, wx = np.argwhere(lw == L).min(0); fy, fx = np.argwhere(lab == L).min(0); y0, x0 = int(fy - wy), int(fx - wx)
assert np.array_equal(lab[y0:y0 + Zs, x0:x0 + Zs], lw), 'window offset not recovered'
print('crlm window', y0, x0)
genes = json.load(open(f'{cd}/genes.json')); T = np.load(f'{cd}/tx.npz')
gy, gx, gidx = T['gy'].astype(np.int64), T['gx'].astype(np.int64), T['gidx'].astype(np.int64)
inw = (gy >= y0) & (gy < y0 + Zs) & (gx >= x0) & (gx < x0 + Zs)
cog, used = [], {}
for p in marks:
    gs = [genes.index(g) for g in OWN.get(p, []) if g in genes]; used[p] = [genes[i] for i in gs]
    k = inw & np.isin(gidx, gs); d = np.zeros((Zs, Zs), np.float32); np.add.at(d, (gy[k] - y0, gx[k] - x0), 1)
    cog.append(gaussian_filter(d, 2.0))
print('coding genes', used)
np.savez_compressed(f'{OUT}/crlm_tumor3_A3_cognate.npz', cognate=np.stack(cog), marks=np.array(marks), window=np.array([y0, x0, Zs]))

M = pd.read_csv(f'{OUT}/metrics_allm.csv'); rows = []
for c, f in [('kidney_rcc', 'results/bench/own/bench_rows.csv'), (case, 'results/bench/own/bench_rows.csv')]:
    R = pd.read_csv(f); R = R[(R.case == c) & (R.method == 'own_mRNA')]
    n_ref = pd.read_csv('results/bench/cellbase_s1/bench_rows.csv').query('case == @c and method == "mlp_pm"').n.max()
    print(c, 'own_mRNA rows', len(R), 'n', R.n.max(), 'vs trained methods', n_ref)
    rows.append(R.assign(method='cognate')[['case', 'marker', 'method', 'spearman', 'pearson', 'auroc']])
pd.concat(rows).to_csv(f'{OUT}/metrics_cognate.csv', index=False)
print('saved')
