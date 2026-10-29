"""Common held-out windows of section rcc in which all four displayed synthetic proteins (immune membrane, proliferation
nucleus, epithelial cytoplasm, induced membrane 40 um) are expressed; every channel of truth, TxCellNet (clean +
randomized, from sim_loc.py predictions), clean-only and the per-cell methods -> results/sim/loc/crops_common_rcc.npz
usage: sim_loc_crops.py   (CPU; after sim_loc.py)"""
import json, numpy as np, pandas as pd
from scipy.ndimage import uniform_filter
SIM, OUT, s = 'data/sim', 'results/sim/loc', 'rcc'
meta = json.load(open(f'{SIM}/{s}/meta.json')); CH = meta['prots']
lab = np.load(f'{SIM}/{s}/cells.npy'); nuc = np.load(f'{SIM}/{s}/nuc.npy'); H, W = lab.shape
edges = np.array(json.load(open(f'results/cell_baseline/{s}_stripes.json'))['x_edges_um']) / meta['um_per_px']
tcol = np.digitize(np.arange(W), edges) % 2 == 1
valid, tissue = np.load(f'{SIM}/{s}/valid.npy'), np.load(f'{SIM}/{s}/tissue.npy'); fitm = valid & tissue & ~tcol[None, :]
ii = np.flatnonzero(fitm.ravel())[np.random.default_rng(0).choice(int(fitm.sum()), 2_000_000)]
Yn = np.load(f'{SIM}/{s}/Y.npy', mmap_mode='r'); Yc = np.load(f'{SIM}/{s}/Yclean.npy', mmap_mode='r')
mu = np.array([np.asarray(Yn[j], np.float32).ravel()[ii].mean() for j in range(6)])
sd = np.array([np.asarray(Yn[j], np.float32).ravel()[ii].std() + 1e-4 for j in range(6)])
Pp = np.load(f'{OUT}/pred_sim_plain_{s}.npy', mmap_mode='r'); Pa = np.load(f'{OUT}/pred_sim_aug_{s}.npy', mmap_mode='r')
truth = pd.read_parquet(f'{SIM}/{s}/cells_truth.parquet')
SHOW = ['mem_immune', 'nuc_prolif', 'cyto_epith', 'niche_r40']; Zw = 128; q = Zw // 8
fr = []
for c in SHOW:
    lv = np.r_[0.0, truth[f'level_{c}'].values]; hi = (lv > np.percentile(lv[1:], 80))[lab]
    s8 = hi[:H // 8 * 8, :W // 8 * 8].reshape(H // 8, 8, W // 8, 8).mean((1, 3)); fr.append(uniform_filter(s8, q, mode='constant'))
score = np.min(fr, 0)
okx = np.array([tcol[x8 * 8:x8 * 8 + Zw].all() for x8 in range(W // 8)])
cand = sorted([(score[y + q // 2, x + q // 2], y * 8, x * 8) for y in range(2, H // 8 - q - 2) for x in range(2, W // 8 - q - 2) if okx[x]], reverse=True)
picked = []
for v, y, x in cand:
    if all(abs(y - a) > Zw or abs(x - b) > Zw for a, b in picked): picked.append((y, x)); print('window', y, x, 'score', round(float(v), 3))
    if len(picked) == 8: break
CP = {m: np.load(f'results/sim/sota/pred_{m}_none.npz')[s].astype(np.float32) for m in ('nichetrans', 'dgat', 'spamosaic', 'totalvi')}
for m in CP: CP[m] = np.where(np.isfinite(CP[m]), CP[m], np.nanmean(CP[m], 0))
out = dict(channels=np.array(CH), ymu=mu, ysd=sd)
for k, (y, x) in enumerate(picked):
    sl = (slice(y, y + Zw), slice(x, x + Zw)); lr = lab[sl]
    out[f'w{k}_yx'] = np.array([y, x]); out[f'w{k}_lab'] = lr; out[f'w{k}_nuc'] = nuc[sl]
    out[f'w{k}_gt'] = np.asarray(Yc[:, sl[0], sl[1]], np.float32)
    zp, za = np.asarray(Pp[:, sl[0], sl[1]], np.float32), np.asarray(Pa[:, sl[0], sl[1]], np.float32)
    out[f'w{k}_tx'] = 0.5 * (zp + za) * sd[:, None, None] + mu[:, None, None]
    out[f'w{k}_clean'] = zp * sd[:, None, None] + mu[:, None, None]
    for m, cp in CP.items(): out[f'w{k}_{m}'] = np.where(lr[None] > 0, cp[lr].transpose(2, 0, 1), np.nan).astype(np.float32)
np.savez_compressed(f'{OUT}/crops_common_rcc.npz', **out); print('saved', len(picked), 'windows')
