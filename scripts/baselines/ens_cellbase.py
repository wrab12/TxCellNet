"""Seed ensembles of the cell-level baselines: average the per-cell predictions saved by baselines_cell.py
(preds/<case>__<method>[_s<k>].npy in the given directories) and score them like run_bench_gen.py.
usage: ens_cellbase.py OUT_DIR METHOD[,METHOD...] DIR[,DIR...]    -> methods <METHOD>_e<n>
"""
import glob, json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp

from txcellnet.bench import combine, log
from txcellnet.metrics import metrics

OUT, METHODS, DIRS = sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(',')
os.makedirs(OUT, exist_ok=True)


rows = []
cases = sorted({os.path.basename(f).split('__')[0] for d in DIRS for f in glob.glob(f'{d}/preds/*.npy')})
for c in cases:
    cd = f'data/cases/{c}'; meta = json.load(open(f'{cd}/meta.json')); cells = pd.read_parquet(f'{cd}/cells.parquet')
    Xc = sp.load_npz(f'{cd}/counts.npz').tocsr()
    ok = (cells['area_px'].values >= 10) & (np.asarray(Xc.sum(1)).ravel() >= 1)
    if 'eval_mask' in meta: ok &= cells[meta['eval_mask']].values.astype(bool)
    for m in METHODS:
        fs = [f for d in DIRS for f in glob.glob(f'{d}/preds/{c}__{m}.npy') + glob.glob(f'{d}/preds/{c}__{m}_s[0-9].npy')]
        if len(fs) < 2: continue
        pc = np.mean([np.load(f).astype(np.float32) for f in fs], 0)
        for key, mk in meta['markers'].items():
            tt = cells[mk['truth']].values.astype(np.float64)
            if mk.get('log', True): tt = np.log1p(np.clip(tt - mk.get('offset', 0), 0, None))
            rows.append(dict(case=c, marker=key, method=f'{m}_e{len(fs)}', **metrics(combine(pc, mk['components'], key)[ok], tt[ok]), n=int(ok.sum())))
    log(c, 'done')
pd.DataFrame(rows).to_csv(f'{OUT}/bench_rows.csv', index=False)
