"""Add all 27 measured proteins as markers of the kidney benchmark cases (truth = per-cell mean of the training target
image log1p(max(x-100,0)) on the same 0.85 um grid; evaluated on held-out stripes only)."""
import json
import numpy as np, pandas as pd
R = 'data/xenium_protein_rcc/pilot_L2'
OWN = json.load(open(f'{R}/meta.json'))['own_genes']
for s in ['rcc', 'ccrcc']:
    cd, sd = f'data/cases/kidney_{s}', f'data/xenium_protein_{s}/pilot_L2'
    prots = json.load(open(f'{sd}/meta.json'))['prots']
    Y = np.load(f'{sd}/Y.npy', mmap_mode='r'); lab = np.load(f'{cd}/lab.npy')
    assert Y.shape[1:] == lab.shape, (Y.shape, lab.shape)
    cells = pd.read_parquet(f'{cd}/cells.parquet'); n = int(lab.max()) + 1; flat = lab.ravel()
    area = np.bincount(flat, minlength=n).astype(np.float64)
    meta = json.load(open(f'{cd}/meta.json'))
    for j, p in enumerate(prots):
        v = np.bincount(flat, weights=np.asarray(Y[j], np.float32).ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
        cells[f'truthprot_{p}'] = v[cells['label'].values]
        meta['markers'][p] = dict(truth=f'truthprot_{p}', components=[p], own=OWN.get(p, []), log=False)
    cells.to_parquet(f'{cd}/cells.parquet'); json.dump(meta, open(f'{cd}/meta.json', 'w'), indent=1)
    print(s, 'added', len(prots), 'protein markers')
