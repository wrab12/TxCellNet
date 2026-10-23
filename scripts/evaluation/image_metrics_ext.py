"""Image-level comparison on the external datasets, against their measured protein images (Y.npy from prep_targets.py:
segmentation stains, IMC or COMET registered to the Xenium grid), on held-out stripes.

Windows: up to K=4 windows of 1024 x 1024 px with the most in-cell held-out pixels (all of the held-out region of small
datasets); each prediction uses a 64-px context margin. Pixels scored: held-out stripe & target mask & inside a cell;
cells with >= 10 such pixels.
Methods
  TxCellNet (within)   the network trained on the dataset's own training stripes (train_case.py), 4 flips
  TxCellNet (cross)    the six kidney-trained networks of the frozen model, 4 flips each; stains = combination of the
                       predicted component proteins with the kidney-fitted weights (as for the cell-level scores)
  NicheTrans / DGAT    per-cell predictions painted into the cells (within: runs/within_baselines; cross: the seed-1
                       NicheTrans of nichetrans_official.py, trained as in its official tutorial, results/bench/nt_off_*_s1)
  cognate mRNA         Gaussian-smoothed (sigma 2 px) density of the marker's coding-gene molecules
Metrics per channel: pixel_r (Pearson over scored pixels), within_r (Pearson of pixel minus its cell's mean: sub-cellular
pattern; 0 by construction for painted per-cell predictions) and enrich (median over cells of boundary-ring mean minus
interior mean, z units) for the measured image and each method.
usage: [ONLY_PAINT=1] image_metrics_ext.py OUT_CSV CASE [CASE ...]   (ONLY_PAINT=1: the painted per-cell methods only)
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F, cv2
from scipy.ndimage import gaussian_filter

from txcellnet.bench import PROTS, OWN as KOWN
from txcellnet.model import GenNet

OUT, CASES = sys.argv[1], sys.argv[2:]; ONLY = os.environ.get('ONLY_PAINT') == '1'
dev = 'cuda'; K, Tt, O = 4, 1024, 64
R0 = 'data/xenium_protein_rcc/pilot_L2/runs'; CROSS = [f'fin_{r}_s{k}' for r in ('clean', 'rand') for k in (1, 2, 3)]
CW = json.load(open('results/bench/cocktail_weights.json'))['mean']


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_net(path_cfg, path_w, chans, own):
    c = json.load(open(path_cfg)); net = GenNet(chans, own, c['cfg'], c['train_genes']).to(dev)
    net.load_state_dict(torch.load(path_w, map_location=dev, weights_only=True), strict=False); net.eval()
    return net, net.prepare(c['train_genes'])


CNETS = [] if ONLY else [load_net(f'{R0}/{t}/cfg.json', f'{R0}/{t}/model.pt', PROTS, KOWN) for t in CROSS]


def predict(net, vocab, genes, T, dapi, lab, win):
    """prediction image (C, h, w) of window win = (y0, x0, y1, x1) with a context margin, 4 flips."""
    H, W = lab.shape; y0, x0, y1, x1 = win; ya, xa, yb, xb = max(y0 - O, 0), max(x0 - O, 0), min(y1 + O, H), min(x1 + O, W)
    gpos = {g: i for i, g in enumerate(vocab)}; cmap = np.array([gpos.get(g, -1) for g in genes])
    m = (T['gy'] >= ya) & (T['gy'] < yb) & (T['gx'] >= xa) & (T['gx'] < xb); g = cmap[T['gidx'][m]]; k = g >= 0
    ly, lx, gg = (torch.from_numpy(a).to(dev) for a in (T['gy'][m][k] - ya, T['gx'][m][k] - xa, g[k]))
    hh, ww = yb - ya + (-(yb - ya)) % 8, xb - xa + (-(xb - xa)) % 8
    x = F.pad(torch.from_numpy(dapi[ya:yb, xa:xb].astype(np.float32))[None, None].to(dev), (0, ww - (xb - xa), 0, hh - (yb - ya)))
    lb = F.pad(torch.from_numpy(lab[ya:yb, xa:xb])[None].to(dev), (0, ww - (xb - xa), 0, hh - (yb - ya)))
    o = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for fl in ([], [-1], [-2], [-1, -2]):
            fy = hh - 1 - ly if -2 in fl else ly; fx = ww - 1 - lx if -1 in fl else lx
            oi = net(x.flip(fl) if fl else x, (fy * ww + fx, gg, 1, hh, ww), lb.flip(fl) if fl else lb)
            o = o + (oi.flip(fl) if fl else oi)
    return (o / 4)[0, :, y0 - ya:y1 - ya, x0 - xa:x1 - xa].float().cpu().numpy()


def geometry(lab):
    b = np.zeros_like(lab, bool)
    b[:-1] |= lab[:-1] != lab[1:]; b[1:] |= lab[1:] != lab[:-1]; b[:, :-1] |= lab[:, :-1] != lab[:, 1:]; b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b &= lab > 0
    dist = cv2.distanceTransform(((lab > 0) & ~b).astype(np.uint8), cv2.DIST_L2, 3)
    return b, (dist >= 2) & (lab > 0)


rows = []
for case in CASES:
    cd = f'data/cases/{case}'; meta = json.load(open(f'{cd}/meta.json')); yj = json.load(open(f'{cd}/Y.json')); chans = yj['channels']
    lab = np.load(f'{cd}/lab.npy'); H, W = lab.shape; dapi = np.load(f'{cd}/dapi.npy')
    Y = np.load(f'{cd}/Y.npy', mmap_mode='r'); ymask = np.load(f'{cd}/ymask.npy')
    edges = np.array(json.load(open(f'{cd}/split.json'))['x_edges_px']); tcol = np.digitize(np.arange(W), edges) % 2 == 1
    score_px = ymask & tcol[None, :] & (lab > 0)
    # windows with the most scored pixels
    cand = [(y, x, min(y + Tt, H), min(x + Tt, W)) for y in range(0, H, Tt) for x in range(0, W, Tt)]
    cand = sorted(cand, key=lambda w: -int(score_px[w[0]:w[2], w[1]:w[3]].sum()))[:K]
    cand = [w for w in cand if score_px[w[0]:w[2], w[1]:w[3]].sum() > 1000]
    keys = [k for k in meta['markers'] if k in yj['markers']]
    cells = pd.read_parquet(f'{cd}/cells.parquet'); labels = cells['label'].values; n = int(lab.max()) + 1
    # per-cell predictions to paint (rows = cells.parquet)
    paint = {}
    for m, f in [('NicheTrans (within)', f'{cd}/runs/within_baselines/nichetrans_off.npy'), ('DGAT (within)', f'{cd}/runs/within_baselines/dgat.npy')]:
        if os.path.exists(f): paint[m] = np.load(f).astype(np.float32)  # columns = keys
    fx = sorted(glob.glob(f'results/bench/nt_off_*_s1/preds/{case}__nichetrans_off.npy'))
    if fx:
        P = np.load(fx[0]).astype(np.float32)  # kidney proteins (z-units)
        comb = []
        for k in keys:
            comps = meta['markers'][k]['components']; w = CW.get(k, {}).get('weights', [1.0] * len(comps))
            comb.append(sum(wi * P[:, PROTS.index(c)] for wi, c in zip(w, comps)))
        paint['NicheTrans (cross)'] = np.stack(comb, 1)
    luts = {}
    for m, P in paint.items():
        lut = np.full((n, len(keys)), np.nan, np.float32); lut[labels] = P; luts[m] = lut
    # within-trained network (reads all panel genes) and the kidney-trained ensemble (reads vocabulary genes)
    wnet = None
    if not ONLY and os.path.exists(f'{cd}/runs/within_clean/model.pt'):
        wnet = load_net(f'{cd}/runs/within_clean/cfg.json', f'{cd}/runs/within_clean/model.pt', chans, {})
    Ta = dict(np.load(f'{cd}/tx_all.npz')); ga = json.load(open(f'{cd}/genes_all.json'))
    Tv = dict(np.load(f'{cd}/tx.npz')); gv = json.load(open(f'{cd}/genes.json'))
    for T_ in (Ta, Tv):
        for a in ('gy', 'gx', 'gidx'): T_[a] = T_[a].astype(np.int64)
    gpos_a = {g: i for i, g in enumerate(ga)}
    acc = {}  # method -> channel -> list of (pred pixels, meas pixels, label pixels, ring flags, core flags)
    for win in cand:
        y0, x0, y1, x1 = win; lw = lab[y0:y1, x0:x1]; sc_ = score_px[y0:y1, x0:x1]
        ring, core = geometry(lw)
        preds = {}
        if wnet is not None:
            pw = predict(wnet[0], wnet[1], ga, Ta, dapi, lab, win)
            preds['TxCellNet (within)'] = [pw[chans.index(yj['markers'][k])] for k in keys]
        if not ONLY:
            pc = np.mean([predict(net, vocab, gv, Tv, dapi, lab, win) for net, vocab in CNETS], 0)
            comb = []
            for k in keys:
                comps = meta['markers'][k]['components']; w = CW.get(k, {}).get('weights', [1.0] * len(comps))
                comb.append(sum(wi * pc[PROTS.index(c)] for wi, c in zip(w, comps)))
            preds['TxCellNet (cross)'] = comb
        for m, lut in luts.items():
            preds[m] = [np.where(lw > 0, lut[lw, j], np.nan) for j in range(len(keys))]
        dens = []  # cognate-mRNA density
        mm = (Ta['gy'] >= y0) & (Ta['gy'] < y1) & (Ta['gx'] >= x0) & (Ta['gx'] < x1)
        for k in keys:
            gi = [gpos_a[g] for g in meta['markers'][k].get('own', []) if g in gpos_a]; d = np.zeros((y1 - y0, x1 - x0), np.float32)
            if gi:
                s_ = mm & np.isin(Ta['gidx'], gi); np.add.at(d, (Ta['gy'][s_] - y0, Ta['gx'][s_] - x0), 1)
            dens.append(gaussian_filter(d, 2.0))
        if not ONLY: preds['cognate mRNA'] = dens
        for j, k in enumerate(keys):
            meas = np.asarray(Y[chans.index(yj['markers'][k]), y0:y1, x0:x1], np.float32)
            for m, pl in preds.items():
                acc.setdefault(m, {}).setdefault(k, []).append((pl[j][sc_], meas[sc_], lw[sc_], ring[sc_], core[sc_]))
    for m, per in acc.items():
        for k, parts in per.items():
            p = np.concatenate([a[0] for a in parts]).astype(np.float64); y = np.concatenate([a[1] for a in parts]).astype(np.float64)
            l = np.concatenate([a[2] + 10_000_000 * i for i, a in enumerate(parts)]); rg = np.concatenate([a[3] for a in parts]); cr = np.concatenate([a[4] for a in parts])
            ok = np.isfinite(p) & np.isfinite(y)
            p, y, l, rg, cr = p[ok], y[ok], l[ok], rg[ok], cr[ok]
            u, li = np.unique(l, return_inverse=True); cnt = np.bincount(li).astype(np.float64); good = cnt[li] >= 10
            pm = np.bincount(li, p) / cnt; ym = np.bincount(li, y) / cnt
            pr, yr = (p - pm[li])[good], (y - ym[li])[good]
            within = float(np.corrcoef(pr, yr)[0, 1]) if pr.std() > 1e-9 else 0.0
            pixel = float(np.corrcoef(p[good], y[good])[0, 1]) if p[good].std() > 1e-9 else np.nan

            def enrich(v):  # median over cells of ring mean - core mean (z units of the window values)
                z = (v - v.mean()) / (v.std() + 1e-9)
                nr = np.bincount(li[rg], minlength=len(u)); nc = np.bincount(li[cr], minlength=len(u))
                er = np.bincount(li[rg], z[rg], minlength=len(u)) / np.maximum(nr, 1); ec = np.bincount(li[cr], z[cr], minlength=len(u)) / np.maximum(nc, 1)
                gc = (nr >= 4) & (nc >= 4)
                return float(np.median((er - ec)[gc])) if gc.any() else np.nan
            rows.append(dict(case=case, method=m, marker=k, n_pixels=int(good.sum()), n_cells=int((cnt >= 10).sum()), pixel_r=pixel, within_r=within,
                             enrich_pred=enrich(p) if p.std() > 1e-9 else 0.0, enrich_meas=enrich(y)))
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    d = df[df.case == case]; log(case, 'windows', len(cand), '\n' + d.groupby('method')[['pixel_r', 'within_r']].median().round(3).to_string())
log('done')
