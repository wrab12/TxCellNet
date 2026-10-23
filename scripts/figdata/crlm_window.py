"""Window arrays of the colorectal liver metastasis example (Fig. 3b; Xenium + same-section IMC; models trained on kidney).

Window: 300 x 300 px (255 um) inside the IMC ROI that contains tumour epithelium, immune cells and stroma: score = min over
PanCK / CD45 / Vimentin of the fraction of window pixels above that marker's ROI-wide 80th percentile.
Arrays (markers PanCK, CD45, Vimentin, CD68, Ki-67): measured IMC (log1p, registered to the Xenium grid), TxCellNet stage 1
(six-network ensemble, 4 flips, 128-px context), stage 2 (refined on the context window: tau 0.85, 20 steps, noise seed
1000), NicheTrans (seed 1, per-cell predictions painted into the cells).
usage: crlm_window.py OUT_DIR [CASE=crlm_tumor3_A3] [IMC_NPZ] [REFINE_RUN=refine_fin]   -> OUT_DIR/crlm_window_<CASE>.npz
"""
import glob, json, os, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F, cv2
from scipy.ndimage import gaussian_filter

from txcellnet.bench import PROTS, OWN
from txcellnet.flow import load_refiner, refine
from txcellnet.model import GenNet

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
CASE = sys.argv[2] if len(sys.argv) > 2 else 'crlm_tumor3_A3'
NPZ = sys.argv[3] if len(sys.argv) > 3 else 'data/crlm_xenium_imc/imc_npz/postxnmIMC_tumor3_M397T__ROI_008-A3.npz'
REFRUN = sys.argv[4] if len(sys.argv) > 4 else 'refine_fin'
cd = f'data/cases/{CASE}'; dev = 'cuda'
meta = json.load(open(f'{cd}/meta.json'))
dapi = np.load(f'{cd}/dapi.npy').astype(np.float32); lab = np.load(f'{cd}/lab.npy'); H, W = lab.shape
genes = json.load(open(f'{cd}/genes.json')); T = np.load(f'{cd}/tx.npz')
gy, gx, gidx = T['gy'].astype(np.int64), T['gx'].astype(np.int64), T['gidx'].astype(np.int64)
MARK = ['PanCK', 'CD45', 'Vimentin', 'CD68', 'Ki-67']

# measured IMC warped onto the grid with the stored affine
Z = np.load(NPZ); chans = [str(c) for c in Z['channels']]; img = Z['img'].astype(np.float32)
M = np.array(meta['affine_imc_px_to_crop_px'])
meas = {m: cv2.warpAffine(img[chans.index(m)], M, (W, H), flags=cv2.INTER_LINEAR) for m in MARK}
roi = cv2.warpAffine(np.ones(img.shape[1:], np.float32), M, (W, H), flags=cv2.INTER_NEAREST) > 0.5

# window: 300 x 300 px (255 um) inside the ROI that contains all three compartments (tumour epithelium, immune, stroma):
# score = min over PanCK / CD45 / Vimentin of the fraction of window pixels above that marker's ROI-wide 80th percentile
Zs = 300; s16 = lambda a: cv2.resize(a, (W // 16, H // 16), interpolation=cv2.INTER_AREA)
inside = s16(roi.astype(np.float32))
pos = {}
for m in ['PanCK', 'CD45', 'Vimentin']:
    a = gaussian_filter(np.log1p(meas[m]), 2); thr = np.percentile(a[roi], 80); pos[m] = s16((a > thr).astype(np.float32))
best, y0, x0 = -1, 0, 0
for y in range(0, H // 16 - Zs // 16):
    for x in range(0, W // 16 - Zs // 16):
        if inside[y:y + Zs // 16, x:x + Zs // 16].min() < 0.99: continue
        v = min(pos[m][y:y + Zs // 16, x:x + Zs // 16].mean() for m in pos)
        if v > best: best, y0, x0 = v, y * 16, x * 16
print('window score (min compartment fraction)', round(float(best), 3))
print('window', y0, x0)

# TxCellNet stage 1 on the window (+128 px context): the six networks of the final ensemble
MEMBERS = [f'fin_{r}_s{k}' for r in ('clean', 'rand') for k in (1, 2, 3)]
R0 = 'data/xenium_protein_rcc/pilot_L2/runs'
Mg = 128; ya, xa, yb, xb = max(y0 - Mg, 0), max(x0 - Mg, 0), min(y0 + Zs + Mg, H), min(x0 + Zs + Mg, W)
hh, ww = yb - ya + (-(yb - ya)) % 8, xb - xa + (-(xb - xa)) % 8
x_in = F.pad(torch.from_numpy(dapi[ya:yb, xa:xb])[None, None].to(dev), (0, ww - (xb - xa), 0, hh - (yb - ya)))
l_in = F.pad(torch.from_numpy(lab[ya:yb, xa:xb])[None].to(dev), (0, ww - (xb - xa), 0, hh - (yb - ya)))
kk = (gy >= ya) & (gy < yb) & (gx >= xa) & (gx < xb)


def run(net, vocab):
    vpos = {g: i for i, g in enumerate(vocab)}; gm = np.array([vpos.get(g, -1) for g in genes]); g = gm[gidx[kk]]; ok = g >= 0
    ly, lx, gg = (torch.from_numpy(a).to(dev) for a in (gy[kk][ok] - ya, gx[kk][ok] - xa, g[ok]))
    o = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for fl in ([], [-1], [-2], [-1, -2]):
            fy = hh - 1 - ly if -2 in fl else ly; fx = ww - 1 - lx if -1 in fl else lx
            oi = net(x_in.flip(fl) if fl else x_in, (fy * ww + fx, gg, 1, hh, ww), l_in.flip(fl) if fl else l_in)
            o = o + (oi.flip(fl) if fl else oi)
    return (o / 4)[0].float().cpu().numpy()  # whole context window


preds = []
for t in MEMBERS:
    c = json.load(open(f'{R0}/{t}/cfg.json'))
    net = GenNet(PROTS, OWN, c['cfg'], c['train_genes']).to(dev)
    net.load_state_dict(torch.load(f'{R0}/{t}/model.pt', map_location=dev, weights_only=True), strict=False); net.eval()
    preds.append(run(net, net.prepare(c['train_genes'])))
ctx = np.mean(preds, 0)[:, :yb - ya, :xb - xa]
ens = ctx[:, y0 - ya:y0 - ya + Zs, x0 - xa:x0 - xa + Zs]
print('ensemble members', len(preds))
# stage 2 on the context window, then cropped
rnet, rsd, steps = load_refiner(f'data/xenium_protein_rcc/pilot_L2/runs/{REFRUN}', device=dev)
tis = (np.load(f'{cd}/tissue.npy')[ya:yb, xa:xb] if os.path.exists(f'{cd}/tissue.npy') else np.ones((yb - ya, xb - xa), bool)).astype(np.float64)  # CRLM cases: no tissue mask
refd = refine(rnet, rsd, ctx, dapi[ya:yb, xa:xb], lab[ya:yb, xa:xb], inside=tis, steps=steps)[:, y0 - ya:y0 - ya + Zs, x0 - xa:x0 - xa + Zs]

# NicheTrans, panel-matched, trained as in its official tutorial: per-cell predictions of nichetrans_official.py (seed 1,
# rows = cells.parquet)
cells = pd.read_parquet(f'{cd}/cells.parquet')
npred = np.load(sorted(glob.glob(f'results/bench/nt_off_*_s1/preds/{CASE}__nichetrans_off.npy'))[0]).astype(np.float32)
lut = np.full((int(lab.max()) + 1, len(PROTS)), np.nan, np.float32); lut[cells['label'].values] = npred
lw = lab[y0:y0 + Zs, x0:x0 + Zs]

np.savez_compressed(f'{OUT}/crlm_window_{CASE}.npz', marks=np.array(MARK), lab=lw,
                    measured=np.stack([np.log1p(meas[m][y0:y0 + Zs, x0:x0 + Zs]) for m in MARK]),
                    stage1=np.stack([ens[PROTS.index(m)] for m in MARK]), stage2=np.stack([refd[PROTS.index(m)] for m in MARK]),
                    nichetrans=np.stack([np.where(lw > 0, lut[lw, PROTS.index(m)], np.nan) for m in MARK]))
print('saved', f'{OUT}/crlm_window_{CASE}.npz')
