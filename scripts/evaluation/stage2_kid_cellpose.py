"""Deep-feature distribution distance for stage 1 / stage 2 vs measured (kidney held-out stripes, 27 proteins x 2 sections).

Feature network: the Cellpose 3 'nuclei' model (pretrained on fluorescence microscopy; cellpose fetches its weights to
~/.cellpose/models on first use), used instead of an ImageNet network. Feature = its style vector (global average of the
deepest encoder map, 256-d, L2-normalized by Cellpose), the network's own whole-image descriptor.
Tiles: the 48 held-out windows of 256 x 256 px of scripts/train/train_refine.py / scripts/evaluation/stage2_realism.py (same random draws), each split into
4 tiles of 128 x 128 px (192 tiles per image), normalized as Cellpose does (1st-99th percentile of the tile -> 0-1),
upsampled x2 (bilinear) so that nuclei (~9 px at 0.85 um/px) approach the model's 17-px training diameter; channel 2 = 0.
KID (Binkowski et al., ICLR 2018): unbiased MMD^2 with the cubic polynomial kernel k(x, y) = (x.y / d + 1)^3 on the style
vectors as the network returns them (metric 'kid_cellpose'); reported x 1e3. Same tiles for all three images.
Variant 'kid_cellpose_cos' (kept for transparency, not used in the figure): style vectors rescaled to norm sqrt(d), so that
x.y / d is the cosine similarity; this kernel saturates (stage 1 and stage 2 both far from measured).
Controls 'noise' / 'noise_s1': stage 1 + matched random noise (stage2_realism.matched_noise, identical images).
usage: [GPU=0] stage2_kid_cellpose.py OUT_CSV   (run from the project root)
"""
import json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from cellpose import models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_realism import matched_noise, CTRL

OUT = sys.argv[1]
SD = {'rcc': 'data/xenium_protein_rcc/pilot_L2', 'ccrcc': 'data/xenium_protein_ccrcc/pilot_L2'}
MEAN_RUN, REF_RUN = 'fin_ens6', 'refine_fin'
prots = json.load(open(f'{SD["rcc"]}/meta.json'))['prots']
dev = torch.device(f'cuda:{os.environ.get("GPU", "0")}')
net = models.CellposeModel(gpu=True, model_type='nuclei', device=dev).net.eval()


def norm99(t):  # cellpose.transforms.normalize99 per tile
    lo, hi = np.percentile(t, 1), np.percentile(t, 99)
    return (t - lo) / (hi - lo) if hi - lo > 1e-3 else np.zeros_like(t)


@torch.no_grad()
def style(tiles):
    x = torch.from_numpy(np.stack([norm99(t) for t in tiles]).astype(np.float32))[:, None].to(dev)
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    x = torch.cat([x, torch.zeros_like(x)], 1)
    out = []
    for i in range(0, len(x), 64):
        _, s, _ = net(x[i:i + 64]); out.append(s.float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def kid(X, Y):
    d = X.shape[1]; k = lambda a, b: (a @ b.T / d + 1) ** 3
    m, n = len(X), len(Y); kxx, kyy, kxy = k(X, X), k(Y, Y), k(X, Y)
    return ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean()) * 1e3


rows = []
for bt, sd in SD.items():
    meta = json.load(open(f'{sd}/meta.json'))
    valid, tissue = np.load(f'{sd}/valid.npy'), np.load(f'{sd}/tissue.npy'); H, W = valid.shape
    edges = np.array(json.load(open(f'results/cell_baseline/{bt}_stripes.json'))['x_edges_um']) / meta['um_per_px']
    test_col = np.digitize(np.arange(W), edges) % 2 == 1
    inside = valid & tissue; fit, test = inside & ~test_col[None, :], inside & test_col[None, :]
    ii = np.flatnonzero(fit.ravel())[np.random.default_rng(0).choice(int(fit.sum()), 2_000_000)]
    rng2 = np.random.default_rng(0); pix = rng2.choice(np.flatnonzero(test.ravel()), 2_000_000, replace=False)  # held-out pixels
    labr = np.load(f'{sd}/cells.npy').ravel(); cnt = np.bincount(labr).astype(np.float64)
    crops = []
    while len(crops) < 48:
        y0, x0 = int(rng2.integers(0, H - 256)), int(rng2.integers(0, W - 256))
        if test[y0:y0 + 256, x0:x0 + 256].mean() > 0.95: crops.append((y0, x0))
    Yr = np.load(f'{sd}/Y.npy', mmap_mode='r'); MU = np.load(f'{sd}/runs/{MEAN_RUN}/pred_full.npy', mmap_mode='r')
    S2 = np.load(f'{sd}/runs/{REF_RUN}/refined_noise0.85.npy', mmap_mode='r')
    for j, p in enumerate(prots):
        t0 = time.time()
        a = np.asarray(Yr[meta['prots'].index(p)], np.float32); v = a.ravel()[ii]
        imgs = dict(measured=((a - v.mean()) / (v.std() + 1e-4)).astype(np.float16).astype(np.float32),
                    stage1=np.asarray(MU[j], np.float32), stage2=np.asarray(S2[j], np.float32))
        for k, sm in CTRL.items(): imgs[k] = matched_noise(imgs['stage1'], imgs['stage2'], labr, cnt, pix, bt, j, sm)[0]
        feats = {k: style([im[y + dy:y + dy + 128, x + dx:x + dx + 128] for y, x in crops for dy in (0, 128) for dx in (0, 128)])
                 for k, im in imgs.items()}
        cos = {k: f / np.linalg.norm(f, axis=1, keepdims=True) * np.sqrt(f.shape[1]) for k, f in feats.items()}
        for metric, F_ in [('kid_cellpose', feats), ('kid_cellpose_cos', cos)]:
            r = dict(section=bt, protein=p, metric=metric, family='deep features (KID)', measured=np.nan,
                     stage1=kid(F_['stage1'], F_['measured']), stage2=kid(F_['stage2'], F_['measured']), n_tiles=len(F_['measured']))
            r.update(dist_stage1=r['stage1'], dist_stage2=r['stage2'])
            for k in CTRL: r[k] = r[f'dist_{k}'] = kid(F_[k], F_['measured'])
            rows.append(r)
            print(time.strftime('%H:%M:%S'), bt, p, metric, f'{time.time() - t0:.0f}s', round(r['stage1'], 4), round(r['stage2'], 4), flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
print(pd.DataFrame(rows).groupby(['metric', 'section'])[['stage1', 'stage2']].median())
