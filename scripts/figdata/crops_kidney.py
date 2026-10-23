"""Crops of a kidney held-out window for the overview / image-level figures: DAPI, transcripts, cell labels, measured
27-plex image, stage-1 mean, stage-2 refined image and painted per-cell baseline predictions.
The window (Z x Z px) is chosen inside a held-out stripe to contain epithelium, immune cells and stroma: maximise the
minimum over E-cadherin / CD45 / Vimentin of the fraction of pixels above that protein's 85th percentile
(window at least 95% inside the held-out tissue).
usage: crops_kidney.py SECTION(rcc|ccrcc) MEAN_RUN|path.npy REFINE_RUN|path.npy OUT_NPZ [Z=240]
"""
import json, sys
import numpy as np, pandas as pd, cv2

sec, MEAN, REF, OUT = sys.argv[1:5]; Z = int(sys.argv[5]) if len(sys.argv) > 5 else 240
sd = f'data/xenium_protein_{sec}/pilot_L2'; case = f'kidney_{sec}'
meta = json.load(open(f'{sd}/meta.json')); prots = meta['prots']
valid, tissue = np.load(f'{sd}/valid.npy'), np.load(f'{sd}/tissue.npy'); H, W = valid.shape
edges = np.array(json.load(open(f'results/cell_baseline/{sec}_stripes.json'))['x_edges_um']) / meta['um_per_px']
test = valid & tissue & (np.digitize(np.arange(W), edges) % 2 == 1)[None, :]
Y = np.load(f'{sd}/Y.npy', mmap_mode='r')
s16 = lambda a: cv2.resize(a.astype(np.float32), (W // 16, H // 16), interpolation=cv2.INTER_AREA)
ins = s16(test)
pos = {}
for p in ['E-Cadherin', 'CD45', 'Vimentin']:
    a = np.asarray(Y[meta['prots'].index(p)], np.float32); thr = np.percentile(a[test], 85); pos[p] = s16(a > thr)
best, y0, x0, k = -1, 0, 0, Z // 16
for y in range(0, H // 16 - k):
    for x in range(0, W // 16 - k):
        if ins[y:y + k, x:x + k].mean() < 0.95: continue
        v = min(pos[p][y:y + k, x:x + k].mean() for p in pos)
        if v > best: best, y0, x0 = v, y * 16, x * 16
print('window', y0, x0, 'score', round(float(best), 3))
sl = (slice(y0, y0 + Z), slice(x0, x0 + Z))
lab = np.load(f'{sd}/cells.npy')[sl]
T = np.load(f'{sd}/tx.npz'); k = (T['gy'] >= y0) & (T['gy'] < y0 + Z) & (T['gx'] >= x0) & (T['gx'] < x0 + Z)
ids = pd.read_csv(f'{sd}/cell_ids.csv')['cell_id'].astype(str).values
cells = pd.read_parquet(f'data/cases/{case}/cells.parquet'); rowlab = pd.Series(np.arange(1, len(ids) + 1), index=ids).reindex(cells['cell_id'].astype(str).values).values
painted = {}
import os
# NicheTrans: trained as in its official tutorial (nichetrans_official.py, panel-matched kidney model, seed 1)
PRED = {'nichetrans_pm': f'results/bench/nt_off_full_s1/preds/{case}__nichetrans_off.npy', 'mlp_pm': f'results/bench/cellbase_s1/preds/{case}__mlp_pm.npy'}
for mn in ['nichetrans_pm', 'mlp_pm']:
    f = PRED[mn]
    if not os.path.exists(f): painted[mn] = np.full((len(prots), Z, Z), np.nan, np.float32); continue  # baseline not run
    P_ = np.load(f).astype(np.float32)
    lut = np.full((len(ids) + 1, len(prots)), np.nan, np.float32); ok = ~np.isnan(rowlab); lut[rowlab[ok].astype(int)] = P_[ok]
    painted[mn] = np.where(lab[None] > 0, lut[lab].transpose(2, 0, 1), np.nan)
np.savez_compressed(OUT, window=np.array([y0, x0, Z]), prots=np.array(prots), genes=np.array(meta['genes']),
                    dapi=np.asarray(np.load(f'{sd}/X.npy', mmap_mode='r')[0][sl], np.float32), lab=lab,
                    ty=T['gy'][k] - y0, tx=T['gx'][k] - x0, tg=T['gid'][k],
                    measured=np.stack([np.asarray(Y[meta['prots'].index(p)][sl], np.float32) for p in prots]),
                    mean=np.asarray(np.load(MEAN if MEAN.endswith('.npy') else f'{sd}/runs/{MEAN}/pred_full.npy', mmap_mode='r')[:, sl[0], sl[1]], np.float32),
                    refined=np.asarray(np.load(REF if REF.endswith('.npy') else f'{sd}/runs/{REF}/refined_noise0.85.npy', mmap_mode='r')[:, sl[0], sl[1]], np.float32),
                    nichetrans=painted['nichetrans_pm'], mlp=painted['mlp_pm'])
print('saved', OUT)
