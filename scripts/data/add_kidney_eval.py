"""Mark held-out-stripe cells in the kidney benchmark cases (same x-stripes as training: digitize(x) % 2 == 1 is test)."""
import json
import numpy as np, pandas as pd
for s in ['rcc', 'ccrcc']:
    cd = f'data/cases/kidney_{s}'
    edges = np.array(json.load(open(f'results/cell_baseline/{s}_stripes.json'))['x_edges_um'])
    cells = pd.read_parquet(f'{cd}/cells.parquet')
    cells['test_stripe'] = (np.digitize(cells['x_um'].values, edges) % 2 == 1)
    cells.to_parquet(f'{cd}/cells.parquet')
    meta = json.load(open(f'{cd}/meta.json')); meta['eval_mask'] = 'test_stripe'
    json.dump(meta, open(f'{cd}/meta.json', 'w'), indent=1)
    print(s, 'test-stripe cells', int(cells['test_stripe'].sum()), 'of', len(cells))
