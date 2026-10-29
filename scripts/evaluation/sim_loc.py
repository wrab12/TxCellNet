"""Semi-synthetic benchmark (sim_make.py): does each synthetic protein appear in the compartment it was placed in?

Every synthetic channel has one intended compartment (the rule of sim_make.py):
  mem_immune, niche_r15, niche_r40, niche_r80   2-px membrane ring of the cell
  nuc_prolif                                    nucleus (vendor nucleus segmentation)
  cyto_epith                                    cytoplasm (cell minus nucleus)
The ground truth is the noiseless blurred image (Yclean); a blurred ring or nucleus also leaks a little signal into the
rest of the cell, so the truth itself does not reach 100 %.

Methods, unperturbed inputs, held-out stripes of both kidney sections, exactly as sim_eval_ens.py / sim_sota.py:
  TxCellNet networks sim_plain (clean), sim_aug (randomized), sim_mix; TxCellNet = ens_cr = mean of clean + randomized
  per-cell methods (NicheTrans official, DGAT, SpaMosaic, totalVI: results/sim/sota/pred_*_none.npz; ridge and
  ridge + neighbourhood refitted as in sim_eval_ens.py), painted uniformly into their cells
Prediction images of TxCellNet are put back on the log(1 + intensity) scale of the targets with the per-section
train-pixel mean and s.d. of the noisy image (train_gen.py normalisation).

Metrics per channel and section over held-out cells with >= 10 held-out pixels (the cells of the cell-level Spearman):
  cell_spearman, pixel_pearson, within_cell   recomputed to check against the stored rows
  share_in    fraction of the cell signal that lies in the intended compartment: sum over compartment pixels of
              max(p - b, 0) / sum over all cell pixels of max(p - b, 0); b = 0 for the truth, b = mean background of the
              noisy training image (E log1p(max(noise, 0)), noise s.d. from the sim_make.py rule) for every prediction,
              since every method was trained on the noisy image. A uniform fill gives the compartment's area fraction.
  share_in_raw  the same with b = 0 for all
  contrast_ratio   sum over cells of (mean in compartment - mean in rest of cell), prediction / truth
  correct_cells    among the expressing cells (truth cell mean above the median of the held-out cells), the fraction whose
                   prediction is higher in the intended compartment than in the rest of the cell (ties = not correct)
Crops: for every channel the best 128 x 128-px windows of section rcc inside a held-out stripe (most expressing cells),
with truth, TxCellNet, clean-only, NicheTrans and DGAT images, cell and nucleus labels -> crops_rcc.npz (inspection only;
the window of Fig. 4a is chosen by scripts/figdata/sim_loc_crops.py)
usage: sim_loc.py [OUT_DIR]        (GPU; results/sim/loc)
"""
import json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from txcellnet.model import GenNet

OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/sim/loc'
os.makedirs(OUT, exist_ok=True)
dev = 'cuda'; torch.manual_seed(0)
SIM = 'data/sim'; SECT = ['rcc', 'ccrcc']; R0 = f'{SIM}/rcc/runs'
CH = json.load(open(f'{SIM}/rcc/meta.json'))['prots']
COMP = {'mem_immune': 'ring', 'nuc_prolif': 'nuc', 'cyto_epith': 'cyto', 'niche_r15': 'ring', 'niche_r40': 'ring', 'niche_r80': 'ring'}
TAGS = ['sim_plain', 'sim_aug', 'sim_mix']


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def stripes(s, W, um):
    edges = np.array(json.load(open(f'results/cell_baseline/{s}_stripes.json'))['x_edges_um']) / um
    return np.digitize(np.arange(W), edges) % 2 == 1  # test columns


def compartments(lab, nuc):  # identical to sim_make.py
    H, W = lab.shape; edge = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0)):
        a, b = lab[:H - dy, :W - dx], lab[dy:, dx:]
        e = (a != b); edge[:H - dy, :W - dx] |= e & (a > 0); edge[dy:, dx:] |= e & (b > 0)
    import cv2
    ring = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool) & (lab > 0)
    return dict(ring=ring, nuc=nuc > 0, cyto=(lab > 0) & (nuc == 0))


def noise_floor(sd):
    """E[log1p(max(x, 0))], x ~ N(0, sd^2): background of the noisy training image where the truth is zero"""
    x = np.random.default_rng(1).normal(0, sd, 2_000_000)
    return float(np.log1p(np.clip(x, 0, None)).mean())


D = {}
for s in SECT:
    sd = f'{SIM}/{s}'; meta = json.load(open(f'{sd}/meta.json'))
    lab = np.load(f'{sd}/cells.npy'); H, W = lab.shape; test_col = stripes(s, W, meta['um_per_px'])
    nuc = np.load(f'{sd}/nuc.npy')
    T = np.load(f'{sd}/tx.npz')
    Yc = np.load(f'{sd}/Yclean.npy', mmap_mode='r'); Yn = np.load(f'{sd}/Y.npy', mmap_mode='r')
    # target normalisation of train_gen.py (Sample): 2M random fit pixels, rng(0)
    valid, tissue = np.load(f'{sd}/valid.npy'), np.load(f'{sd}/tissue.npy')
    fitm = valid & tissue & ~test_col[None, :]
    ii = np.flatnonzero(fitm.ravel())[np.random.default_rng(0).choice(int(fitm.sum()), 2_000_000)]
    ymu, ysd, bg = [], [], []
    for j in range(len(CH)):
        a = np.asarray(Yn[j], np.float32).ravel()[ii]; ymu.append(float(a.mean())); ysd.append(float(a.std()) + 1e-4)
        img = np.expm1(np.asarray(Yc[j], np.float32)[lab > 0])  # noiseless blurred image before log1p
        bg.append(noise_floor(0.05 * np.percentile(img, 99)))
    D[s] = dict(meta=meta, lab=lab, nuc=nuc, test_col=test_col, gy=T['gy'].astype(np.int64), gx=T['gx'].astype(np.int64),
                gid=T['gid'].astype(np.int64), H=H, W=W, Yc=Yc, Yn=Yn, dapi=np.load(f'{sd}/X.npy', mmap_mode='r'),
                ymu=np.array(ymu), ysd=np.array(ysd), bg=np.array(bg), comp=compartments(lab, nuc))
    log(s, 'grid', H, W, 'ymu', np.round(ymu, 3), 'ysd', np.round(ysd, 3), 'background', np.round(bg, 4))


# ---------------- TxCellNet predictions (inference code of sim_eval_ens.py) ----------------
def predict(tag, s):
    f = f'{OUT}/pred_{tag}_{s}.npy'
    if os.path.exists(f): return np.load(f)
    c = json.load(open(f'{R0}/{tag}/cfg.json'))
    net = GenNet(CH, {}, c['cfg'], c['train_genes']).to(dev)
    net.load_state_dict(torch.load(f'{R0}/{tag}/model.pt', map_location=dev, weights_only=True), strict=False); net.eval()
    vocab = net.prepare(c['train_genes']); vpos = {g: i for i, g in enumerate(vocab)}
    d = D[s]; H, W = d['H'], d['W']
    gmap = np.array([vpos.get(g, -1) for g in d['meta']['genes']], np.int64); g = gmap[d['gid']]; k = g >= 0
    ty, tx_, tg = (torch.from_numpy(a[k]).to(dev) for a in (d['gy'], d['gx'], g))
    xd = np.asarray(d['dapi'][0], np.float32); fit = ~d['test_col'][None, :].repeat(H, 0) & (d['lab'] > 0)
    X = torch.from_numpy((xd - xd[fit].mean()) / (xd[fit].std() + 1e-4)).to(dev)
    Lt = torch.from_numpy(d['lab']).to(dev)
    pred = np.zeros((len(CH), H, W), np.float16); Tt, O = 1024, 64
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for y0 in range(0, H, Tt - 2 * O):
            for x0 in range(0, W, Tt - 2 * O):
                ya, xa = max(y0 - O, 0), max(x0 - O, 0); yb, xb = min(ya + Tt, H), min(xa + Tt, W)
                hh, ww = yb - ya + (-(yb - ya)) % 8, xb - xa + (-(xb - xa)) % 8
                x = F.pad(X[ya:yb, xa:xb][None, None], (0, ww - (xb - xa), 0, hh - (yb - ya)))
                lb = F.pad(Lt[ya:yb, xa:xb][None], (0, ww - (xb - xa), 0, hh - (yb - ya)))
                m = (ty >= ya) & (ty < yb) & (tx_ >= xa) & (tx_ < xb)
                ly, lx, gg = ty[m] - ya, tx_[m] - xa, tg[m]
                o = 0
                for fl in ([], [-1], [-2], [-1, -2]):
                    fy = hh - 1 - ly if -2 in fl else ly; fx = ww - 1 - lx if -1 in fl else lx
                    oi = net(x.flip(fl) if fl else x, (fy * ww + fx, gg, 1, hh, ww), lb.flip(fl) if fl else lb)
                    o = o + (oi.flip(fl) if fl else oi)
                o = (o / 4)[0, :, :yb - ya, :xb - xa].float()
                ye, xe = min(y0 + Tt - 2 * O, H), min(x0 + Tt - 2 * O, W)
                pred[:, y0:ye, x0:xe] = o[:, y0 - ya:y0 - ya + ye - y0, x0 - xa:x0 - xa + xe - x0].cpu().numpy()
    np.save(f, pred); log(tag, s, 'predicted')
    return pred


# ---------------- per-cell baselines: ridge, ridge + neighbourhood (sim_eval_ens.py) ----------------
def cell_feats(s):
    d = D[s]; L = d['lab'][d['gy'], d['gx']]; n = int(d['lab'].max()) + 1; ng = len(d['meta']['genes'])
    Xc = sp.csr_matrix((np.ones(len(L), np.float32), (L, d['gid'])), shape=(n, ng))
    tot = np.asarray(Xc.sum(1)).ravel()
    return np.log1p((sp.diags(1e3 / np.maximum(tot, 1)) @ Xc).toarray()).astype(np.float32), n


UNION = sorted(set(D['rcc']['meta']['genes']) | set(D['ccrcc']['meta']['genes'])); up = {g: i for i, g in enumerate(UNION)}
FE, TR, TG, NB = {}, {}, {}, {}
for s in SECT:
    d = D[s]; Fe0, n = cell_feats(s)
    Fe = np.zeros((n, len(UNION)), np.float32); Fe[:, [up[g] for g in d['meta']['genes']]] = Fe0
    yy, xx = np.nonzero(d['lab']); L = d['lab'][yy, xx]; cnt = np.bincount(L, minlength=n).astype(np.float64)
    cen = np.stack([np.bincount(L, weights=xx, minlength=n), np.bincount(L, weights=yy, minlength=n)], 1) / np.maximum(cnt, 1)[:, None]
    tgt = np.stack([np.bincount(L, weights=np.asarray(d['Yn'][j], np.float32)[yy, xx].astype(np.float64), minlength=n) / np.maximum(cnt, 1)
                    for j in range(len(CH))], 1)
    train = (cnt > 0) & ~d['test_col'][np.clip(cen[:, 0].astype(int), 0, d['W'] - 1)]; train[0] = False
    FE[s], TR[s], TG[s] = Fe, train, tgt.astype(np.float32)
    NB[s] = NearestNeighbors(n_neighbors=13).fit(cen).kneighbors(cen, return_distance=False)[:, 1:]


def ridge(Xtr, Ytr, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Z = (Xtr - mu) / sd
    Wt = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Ytr - Ytr.mean(0)))
    return lambda X: ((X - mu) / sd) @ Wt + Ytr.mean(0)


CELLP = {s: {} for s in SECT}
for name, niche in [('ridge', False), ('ridge_niche', True)]:
    fx = (lambda s: np.c_[FE[s], FE[s][NB[s]].mean(1)]) if niche else (lambda s: FE[s])
    f = ridge(np.concatenate([fx(s)[TR[s]] for s in SECT]), np.concatenate([TG[s][TR[s]] for s in SECT]))
    for s in SECT: CELLP[s][name] = f(fx(s)).astype(np.float32)
for m, nm in (('nichetrans', 'NicheTrans'), ('dgat', 'DGAT'), ('spamosaic', 'SpaMosaic'), ('totalvi', 'totalVI')):
    P = np.load(f'results/sim/sota/pred_{m}_none.npz')
    for s in SECT:
        cp = P[s].astype(np.float32); CELLP[s][nm] = np.where(np.isfinite(cp), cp, np.nanmean(cp, 0))
log('per-cell predictions ready')


# ---------------- metrics ----------------
def metrics(name, s, j, p_img=None, cell_pred=None, b=0.0):
    """p_img: (H, W) prediction on the log scale; cell_pred: (n_labels,) painted through the labels"""
    d = D[s]; lab = d['lab']; tc = d['test_col'][None, :] & (lab > 0); l = lab[tc]; n = int(lab.max()) + 1
    cnt = np.bincount(l, minlength=n).astype(np.float64); ok = cnt >= 10; test_cells = np.flatnonzero(ok)
    y = np.asarray(d['Yc'][j], np.float32)[tc].astype(np.float64)
    p = (p_img[tc] if p_img is not None else cell_pred[l]).astype(np.float64)
    pm = np.bincount(l, weights=p, minlength=n) / np.maximum(cnt, 1); ym = np.bincount(l, weights=y, minlength=n) / np.maximum(cnt, 1)
    pr, yr = p - pm[l], y - ym[l]
    within = float(np.corrcoef(pr, yr)[0, 1]) if pr.std() > 1e-9 else 0.0
    inside = d['comp'][COMP[CH[j]]][tc]; sel = ok[l]
    share = lambda v: float(v[sel & inside].sum() / v[sel].sum())
    # per-cell compartment means
    nin = np.bincount(l[inside], minlength=n).astype(np.float64); nout = cnt - nin
    both = ok & (nin >= 3) & (nout >= 3)
    def io(v):
        si = np.bincount(l[inside], weights=v[inside], minlength=n); so = np.bincount(l[~inside], weights=v[~inside], minlength=n)
        return si / np.maximum(nin, 1) - so / np.maximum(nout, 1)
    cp, cy = io(p), io(y)
    expr = both & (ym > np.median(ym[test_cells]))
    return dict(method=name, section=s, channel=CH[j], compartment=COMP[CH[j]],
                cell_spearman=float(spearmanr(pm[test_cells], ym[test_cells])[0]), pixel_pearson=float(np.corrcoef(p, y)[0, 1]),
                within_cell=within, share_in=share(np.clip(p - b, 0, None)), share_in_raw=share(np.clip(p, 0, None)),
                contrast_ratio=float(cp[both].sum() / cy[both].sum()), correct_cells=float((cp[expr] > 1e-6).mean()),
                truth_correct_cells=float((cy[expr] > 1e-6).mean()), area_share=float(nin[ok].sum() / cnt[ok].sum()),
                n_cells=int(ok.sum()), n_expr=int(expr.sum()), background=b)


rows, CROPS = [], {}
for s in SECT:
    d = D[s]
    Z = {t: predict(t, s) for t in TAGS}
    Z['ens_cr'] = None
    for j, c in enumerate(CH):
        y = np.asarray(d['Yc'][j], np.float32)
        rows.append(metrics('truth', s, j, p_img=y, b=0.0))
        for t in TAGS + ['ens_cr']:
            z = (0.5 * (Z['sim_plain'][j].astype(np.float32) + Z['sim_aug'][j].astype(np.float32))) if t == 'ens_cr' else Z[t][j].astype(np.float32)
            rows.append(metrics(f'TxCellNet:{t}', s, j, p_img=z * d['ysd'][j] + d['ymu'][j], b=d['bg'][j]))
        for nm, cp in CELLP[s].items():
            rows.append(metrics(nm, s, j, cell_pred=cp[:, j], b=d['bg'][j]))
        log(s, c, 'scored')
    if s == 'rcc':  # crops for the figure
        lab = d['lab']; tcol = d['test_col']; H, W = d['H'], d['W']; Zw = 128
        ens = 0.5 * (Z['sim_plain'].astype(np.float32) + Z['sim_aug'].astype(np.float32))
        nt = CELLP[s]['NicheTrans']; dg = CELLP[s]['DGAT']
        truth = pd.read_parquet(f'{SIM}/{s}/cells_truth.parquet')
        for j, c in enumerate(CH):
            lvl = np.r_[0.0, truth[f'level_{c}'].values]; hi = lvl > np.percentile(lvl[1:], 80)
            expr_px = hi[lab]
            s8 = expr_px[:H // 8 * 8, :W // 8 * 8].reshape(H // 8, 8, W // 8, 8).mean((1, 3))
            from scipy.ndimage import uniform_filter
            q = Zw // 8; box = uniform_filter(s8, q, mode='constant')  # box mean centred; window start = centre - q // 2
            okx = np.array([tcol[x8 * 8:x8 * 8 + Zw].all() for x8 in range(W // 8)])
            cand = [(box[yy + q // 2, xx + q // 2], yy * 8, xx * 8) for yy in range(2, H // 8 - q - 2, 2)
                    for xx in range(2, W // 8 - q - 2, 2) if okx[xx]]
            cand.sort(reverse=True); picked = []
            for v, yy, xx in cand:
                if all(abs(yy - a) > Zw or abs(xx - b_) > Zw for a, b_ in picked): picked.append((yy, xx))
                if len(picked) == 4: break
            for k, (yy, xx) in enumerate(picked):
                sl = (slice(yy, yy + Zw), slice(xx, xx + Zw)); lr = lab[sl]
                CROPS[f'{c}_{k}_yx'] = np.array([yy, xx]); CROPS[f'{c}_{k}_lab'] = lr; CROPS[f'{c}_{k}_nuc'] = d['nuc'][sl]
                CROPS[f'{c}_{k}_gt'] = np.asarray(d['Yc'][j][sl], np.float32)
                CROPS[f'{c}_{k}_tx'] = ens[j][sl] * d['ysd'][j] + d['ymu'][j]
                CROPS[f'{c}_{k}_clean'] = Z['sim_plain'][j][sl].astype(np.float32) * d['ysd'][j] + d['ymu'][j]
                CROPS[f'{c}_{k}_nt'] = np.where(lr > 0, nt[lr, j], np.nan).astype(np.float32)
                CROPS[f'{c}_{k}_dgat'] = np.where(lr > 0, dg[lr, j], np.nan).astype(np.float32)
        CROPS['bg'] = d['bg']
        np.savez_compressed(f'{OUT}/crops_rcc.npz', **CROPS); log('crops saved')
    del Z

df = pd.DataFrame(rows); df.to_csv(f'{OUT}/sim_loc_rows.csv', index=False)
pd.set_option('display.width', 250)
for met in ['cell_spearman', 'within_cell', 'share_in', 'share_in_raw', 'contrast_ratio', 'correct_cells']:
    log(f'\n{met} (mean over the two sections)\n' + df.groupby(['method', 'channel'])[met].mean().unstack('channel')[CH].round(3).to_string())
log('area share\n' + df[df.method == 'truth'].groupby('channel').area_share.mean().round(3).to_string())
