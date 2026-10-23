"""Cognate-mRNA reference (training-free: log1p of the summed counts of the marker's coding genes, library-size normalised)
and per-section statistics (median vocabulary molecules per scored cell, median counts per cell).
usage: own_only.py OUT_DIR CASE_DIR [CASE_DIR ...]   -> OUT_DIR/bench_rows.csv, bench_cases.csv
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp

from txcellnet.io import load_case_image, label_counts
from txcellnet.metrics import metrics
from txcellnet.utils import log

OUT, CASES = sys.argv[1], sys.argv[2:]; os.makedirs(OUT, exist_ok=True)
rows, info = [], []
for cd in CASES:
    case = load_case_image(cd); meta = case['meta']; name = meta['dataset']; cells = case['cells']; labels = cells['label'].values
    Xc = sp.load_npz(f'{cd}/counts.npz').tocsr(); cg = json.load(open(f'{cd}/counts_genes.json'))
    ok = (cells['area_px'].values >= 10) & (np.asarray(Xc.sum(1)).ravel() >= 1)
    if 'eval_mask' in meta: ok &= cells[meta['eval_mask']].values.astype(bool)
    lc = label_counts(case['lab'])
    per_cell = np.bincount(case['lab'][case['gy'], case['gx']], minlength=len(lc))[labels]
    info.append(dict(case=name, platform=meta['platform'], tissue=meta['tissue'], panel=meta.get('panel'), n_cells_eval=int(ok.sum()),
                     vocab_genes_present=len(case['genes']), median_vocab_tx_per_cell=float(np.median(per_cell[ok])),
                     median_counts_per_cell=float(np.median(np.asarray(Xc.sum(1)).ravel()[ok]))))
    tot = np.asarray(Xc.sum(1)).ravel()
    for key, mk in meta['markers'].items():
        idx = [cg.index(g) for g in mk.get('own', []) if g in cg]
        if not idx: continue
        own = np.log1p(np.asarray(Xc[:, idx].sum(1)).ravel() / np.maximum(tot, 1) * np.median(tot[tot > 0]))
        t = cells[mk['truth']].values.astype(np.float64)
        if mk.get('log', True): t = np.log1p(np.clip(t - mk.get('offset', 0), 0, None))
        rows.append(dict(case=name, marker=key, method='own_mRNA', **metrics(own[ok], t[ok]), n=int(ok.sum()), note='+'.join(cg[i] for i in idx)))
    log(name, info[-1], 'own_mRNA median', round(pd.DataFrame(rows).query('case == @name').spearman.median(), 3))
    pd.DataFrame(rows).to_csv(f'{OUT}/bench_rows.csv', index=False); pd.DataFrame(info).to_csv(f'{OUT}/bench_cases.csv', index=False)
