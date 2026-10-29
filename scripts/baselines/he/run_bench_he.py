"""Score the H&E -> protein baseline (train_he.py) on prepared cases that have a same-section H&E (he.npy), with the
same cells, markers and metrics as run_bench_gen.py. H&E normalisation per case: per-channel z-score over tissue pixels.
The last line reports the number of Supplementary Section S5: the mean over the test sections with an H&E (the nine breast
test sections and the lymph node) of the median Spearman over markers.
usage: run_bench_he.py OUT_DIR RUN_TAG CASE_DIR [CASE_DIR ...]
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, torch, torch.nn.functional as F

from txcellnet.bench import PROTS, TEST, combine, log
from txcellnet.metrics import metrics
from he_model import HENet

OUT, TAG, CASES = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(OUT, exist_ok=True)
dev = 'cuda'
R = 'data/xenium_protein_rcc/pilot_L2'
CACHE = 'results/bench/predcache'; os.makedirs(CACHE, exist_ok=True)
net = HENet(len(PROTS)).to(dev); net.load_state_dict(torch.load(f'{R}/runs/{TAG}/model.pt', map_location=dev, weights_only=True)); net.eval()


rows = []
for cd in CASES:
    if not os.path.exists(f'{cd}/he.npy'): continue
    meta = json.load(open(f'{cd}/meta.json')); name = meta['dataset']
    cells = pd.read_parquet(f'{cd}/cells.parquet'); labels = cells['label'].values
    cf = f'{CACHE}/{name}__{TAG}.npy'
    if os.path.exists(cf):
        pc = np.load(cf).astype(np.float64)
    else:
        he = np.load(f'{cd}/he.npy').astype(np.float32); tis = np.load(f'{cd}/tissue.npy').astype(bool)
        # per-channel stats over 2M sampled tissue pixels, exactly as in train_he.py (float64 accumulation)
        ii = np.flatnonzero(tis.ravel()); ii = ii[np.random.default_rng(0).choice(len(ii), min(len(ii), 2_000_000), replace=False)]
        mu = np.array([he[c].ravel()[ii].astype(np.float64).mean() for c in range(3)], np.float32)[:, None, None]
        sd = np.array([he[c].ravel()[ii].astype(np.float64).std() for c in range(3)], np.float32)[:, None, None] + 1e-4
        log(name, 'H&E norm mu', mu.ravel().round(3), 'sd', sd.ravel().round(3))
        X = torch.from_numpy((he - mu) / sd).to(dev); Lt = torch.from_numpy(np.load(f'{cd}/lab.npy')).to(dev)
        H, W = Lt.shape; n = int(Lt.max()) + 1
        csum = torch.zeros(n, len(PROTS), dtype=torch.float64, device=dev)
        FL = [[], [-1], [-2], [-1, -2]]; Tt, O = 1024, 64
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for y0 in range(0, H, Tt - 2 * O):
                for x0 in range(0, W, Tt - 2 * O):
                    ya, xa = max(y0 - O, 0), max(x0 - O, 0); yb, xb = min(ya + Tt, H), min(xa + Tt, W)
                    hh, ww = yb - ya + (-(yb - ya)) % 8, xb - xa + (-(xb - xa)) % 8
                    x = F.pad(X[:, ya:yb, xa:xb][None], (0, ww - (xb - xa), 0, hh - (yb - ya)))
                    lab = F.pad(Lt[ya:yb, xa:xb][None], (0, ww - (xb - xa), 0, hh - (yb - ya)))
                    acc = sum((net(x.flip(f), lab.flip(f)).flip(f) if f else net(x, lab)) for f in FL) / len(FL)
                    oo = acc[0, :, :yb - ya, :xb - xa].float()
                    ye, xe = min(y0 + Tt - 2 * O, H), min(x0 + Tt - 2 * O, W)
                    oc = oo[:, y0 - ya:y0 - ya + ye - y0, x0 - xa:x0 - xa + xe - x0]
                    csum.index_add_(0, Lt[y0:ye, x0:xe].reshape(-1).long(), oc.reshape(len(PROTS), -1).T.double())
        cnt = np.bincount(np.load(f'{cd}/lab.npy').ravel(), minlength=n).astype(np.float64)
        pc = (csum.cpu().numpy() / np.maximum(cnt, 1)[:, None])[labels]
        np.save(cf, pc.astype(np.float16))
    Xc = sp.load_npz(f'{cd}/counts.npz').tocsr()
    ok = (cells['area_px'].values >= 10) & (np.asarray(Xc.sum(1)).ravel() >= 1)
    if 'eval_mask' in meta: ok &= cells[meta['eval_mask']].values.astype(bool)
    for key, mk in meta['markers'].items():
        tt = cells[mk['truth']].values.astype(np.float64)
        if mk.get('log', True): tt = np.log1p(np.clip(tt - mk.get('offset', 0), 0, None))
        rows.append(dict(case=name, marker=key, method=f'HE:{TAG}', **metrics(combine(pc, mk['components'], key)[ok], tt[ok]), n=int(ok.sum())))
    df = pd.DataFrame(rows); df.to_csv(f'{OUT}/bench_rows.csv', index=False)
    log(name, f'HE:{TAG} median Spearman', round(df[df.case == name].spearman.median(), 3))
if rows:  # Supplementary Section S5: mean over the test sections with an H&E of the median Spearman over markers
    med = pd.DataFrame(rows).groupby('case').spearman.median(); test = [c for c in TEST if c in med.index]
    if test: log(f'HE:{TAG} mean over the {len(test)} test sections with H&E', round(float(med[test].mean()), 3))
log('done')
