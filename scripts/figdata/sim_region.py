"""Region arrays of the semi-synthetic example (Fig. 4a): a 200 x 200 px (170 um) held-out window of section rcc with the
most structure in the nuclear, membrane and niche-induced (40 um) channels; ground truth (noiseless image), TxCellNet
(mean of the clean and the randomized network, 4 flips, 128-px context, on the log-intensity scale of the targets),
DAPI (display-scaled), cell boundaries and labels.
usage: sim_region.py OUT_DIR   -> OUT_DIR/sim_region.npz
"""
import json, os, sys
import numpy as np, torch
from skimage.segmentation import find_boundaries

from txcellnet.model import GenNet

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
dev = 'cuda'; torch.manual_seed(0)
SIM = 'data/sim'; S0 = 'rcc'
CH = json.load(open(f'{SIM}/rcc/meta.json'))['prots']


def stripes(s, W, um):
    edges = np.array(json.load(open(f'results/cell_baseline/{s}_stripes.json'))['x_edges_um']) / um
    return np.digitize(np.arange(W), edges) % 2 == 1


# ---------------- data of the figure section ----------------
meta = json.load(open(f'{SIM}/{S0}/meta.json'))
lab = np.load(f'{SIM}/{S0}/cells.npy'); H, W = lab.shape; test_col = stripes(S0, W, meta['um_per_px'])
Yc = np.load(f'{SIM}/{S0}/Yclean.npy', mmap_mode='r'); Yn = np.load(f'{SIM}/{S0}/Y.npy', mmap_mode='r')
dapi = np.asarray(np.load(f'{SIM}/{S0}/X.npy', mmap_mode='r')[0], np.float32)
T = np.load(f'{SIM}/{S0}/tx.npz'); gy, gx, gid = T['gy'].astype(np.int64), T['gx'].astype(np.int64), T['gid'].astype(np.int64)

# region: 200 x 200 px (170 um) window inside a test stripe with the most structure in the nuclear + membrane channels
Z = 200
s8 = lambda a: a[:H // 8 * 8, :W // 8 * 8].reshape(H // 8, 8, W // 8, 8).mean((1, 3))
score = s8(np.asarray(Yc[CH.index('nuc_prolif')], np.float32)) + s8(np.asarray(Yc[CH.index('mem_immune')], np.float32)) + s8(np.asarray(Yc[CH.index('niche_r40')], np.float32))
best, by, bx = -1, 0, 0
for y in range(4, H // 8 - Z // 8 - 4, 3):
    for x in range(4, W // 8 - Z // 8 - 4, 3):
        if not test_col[x * 8:x * 8 + Z].all(): continue
        v = score[y:y + Z // 8, x:x + Z // 8].mean() + 0.5 * score[y:y + Z // 8, x:x + Z // 8].std()
        if v > best: best, by, bx = v, y * 8, x * 8
y0, x0 = by, bx
print('region', y0, x0)


# ---------------- TxCellNet prediction on the region (128 px context) ----------------
def tx_predict(tag):
    c = json.load(open(f'{SIM}/rcc/runs/{tag}/cfg.json'))
    net = GenNet(CH, {}, c['cfg'], c['train_genes']).to(dev)
    net.load_state_dict(torch.load(f'{SIM}/rcc/runs/{tag}/model.pt', map_location=dev, weights_only=True), strict=False); net.eval()
    vocab = net.prepare(c['train_genes']); vpos = {g: i for i, g in enumerate(vocab)}
    gmap = np.array([vpos.get(g, -1) for g in meta['genes']], np.int64)
    M = 128; ya, xa, yb, xb = y0 - M, x0 - M, y0 + Z + M, x0 + Z + M
    k = (gy >= ya) & (gy < yb) & (gx >= xa) & (gx < xb) & (gmap[gid] >= 0)
    ly, lx, g = (torch.from_numpy(a).to(dev) for a in (gy[k] - ya, gx[k] - xa, gmap[gid[k]]))
    fit = ~test_col[None, :].repeat(H, 0) & (lab > 0)
    xd = (dapi - dapi[fit].mean()) / (dapi[fit].std() + 1e-4)
    x = torch.from_numpy(xd[ya:yb, xa:xb])[None, None].to(dev); lb = torch.from_numpy(lab[ya:yb, xa:xb])[None].to(dev)
    hh, ww = yb - ya, xb - xa; o = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for fl in ([], [-1], [-2], [-1, -2]):
            fy = hh - 1 - ly if -2 in fl else ly; fx = ww - 1 - lx if -1 in fl else lx
            oi = net(x.flip(fl) if fl else x, (fy * ww + fx, g, 1, hh, ww), lb.flip(fl) if fl else lb)
            o = o + (oi.flip(fl) if fl else oi)
    z = (o / 4)[0, :, M:M + Z, M:M + Z].float().cpu().numpy()
    # back to the log-intensity scale of the targets (train-stripe pixel statistics of the noisy image)
    out = np.empty_like(z)
    for j in range(len(CH)):
        v = np.asarray(Yn[j], np.float32)[fit][::50]; out[j] = z[j] * v.std() + v.mean()
    return out


pred_tx = 0.5 * (tx_predict('sim_plain') + tx_predict('sim_aug'))  # TxCellNet: clean + randomized networks
lab_r = lab[y0:y0 + Z, x0:x0 + Z]
gt = np.asarray(Yc[:, y0:y0 + Z, x0:x0 + Z], np.float32)
d_r = dapi[y0:y0 + Z, x0:x0 + Z]; d_r = np.clip((d_r - np.percentile(d_r, 1)) / (np.percentile(d_r, 99.5) - np.percentile(d_r, 1)), 0, 1)
bnd = find_boundaries(lab_r, mode='inner')
np.savez_compressed(f'{OUT}/sim_region.npz', pred_tx=pred_tx, gt=gt, d_r=d_r, bnd=bnd, lab_r=lab_r)
print('saved', f'{OUT}/sim_region.npz')
