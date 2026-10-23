"""Evaluate methods on the semi-synthetic benchmark (sim_make.py), held-out stripes of both kidney sections.

Methods
  TxCellNet runs (train_gen.py with SIMROOT=data/sim): pixel images
  cell-level baselines trained on the per-cell mean of the noisy synthetic image (train-stripe cells, both sections):
    ridge (own cell RNA), ridge_niche (+ mean RNA of the 12 nearest cells), NicheTrans (official model class, 12 NN);
    their per-cell predictions are painted uniformly into the cell masks to form images
Metrics per synthetic channel (against the NOISELESS ground truth)
  cell_spearman   per-cell mean prediction vs per-cell truth
  pixel_pearson   all pixels inside cells
  within_cell     Pearson after subtracting each cell's own mean from prediction and truth (localisation only;
                  exactly 0 for any per-cell method)
usage: sim_eval.py OUT_DIR RUN_TAG[,RUN_TAG...] [PERTURB]      PERTURB in {none, thin50, thin25, dilate2, merge20}
"""
import json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, scipy.sparse as sp, h5py
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from txcellnet.model import GenNet

sys.path.insert(0, os.environ.get('NICHETRANS_DIR', 'external/NicheTrans'))
from model.nicheTrans import NicheTrans

OUT, TAGS = sys.argv[1], [t for t in sys.argv[2].split(',') if t]
PERT = sys.argv[3] if len(sys.argv) > 3 else 'none'
os.makedirs(OUT, exist_ok=True)
dev = 'cuda'; torch.manual_seed(0); rng = np.random.default_rng(0)
SIM = 'data/sim'; SECT = ['rcc', 'ccrcc']
CH = json.load(open(f'{SIM}/rcc/meta.json'))['prots']


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def stripes(s, W, um):
    edges = np.array(json.load(open(f'results/cell_baseline/{s}_stripes.json'))['x_edges_um']) / um
    return np.digitize(np.arange(W), edges) % 2 == 1  # test columns


def perturb_labels(lab):
    if PERT == 'dilate2':  # boundaries pushed 2 px (1.7 um) outward into background
        import cv2
        out = lab.copy()
        for _ in range(2):
            d = cv2.dilate(out.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
            out = np.where(out == 0, d, out)
        return out
    if PERT == 'merge20':  # 20% of cells merged into a touching neighbour (under-segmentation)
        H, W = lab.shape
        a, b = lab[:, :-1].ravel(), lab[:, 1:].ravel(); k = (a != b) & (a > 0) & (b > 0)
        pairs = np.unique(np.stack([a[k], b[k]], 1), axis=0)
        sel = pairs[rng.random(len(pairs)) < 0.2]
        mp = np.arange(lab.max() + 1)
        for x, y in sel: mp[y] = mp[x]
        return mp[lab]
    return lab


# ---------------- data per section ----------------
D = {}
for s in SECT:
    sd = f'{SIM}/{s}'; meta = json.load(open(f'{sd}/meta.json'))
    lab_true = np.load(f'{sd}/cells.npy'); H, W = lab_true.shape
    test_col = stripes(s, W, meta['um_per_px'])
    lab_in = perturb_labels(lab_true)
    T = np.load(f'{sd}/tx.npz'); gy, gx, gid = T['gy'].astype(np.int64), T['gx'].astype(np.int64), T['gid'].astype(np.int64)
    if PERT.startswith('thin'):
        keep = rng.random(len(gid)) < int(PERT[4:]) / 100; gy, gx, gid = gy[keep], gx[keep], gid[keep]
    truth = pd.read_parquet(f'{sd}/cells_truth.parquet')
    D[s] = dict(meta=meta, lab=lab_true, lab_in=lab_in, test_col=test_col, gy=gy, gx=gx, gid=gid, truth=truth, H=H, W=W,
                Yc=np.load(f'{sd}/Yclean.npy', mmap_mode='r'), dapi=np.load(f'{sd}/X.npy', mmap_mode='r'))
    log(s, 'grid', H, W, 'molecules', len(gid), 'perturbation', PERT)


def cell_counts(s):
    """Per-cell gene counts from the (possibly perturbed) label map and molecules -> csr (n_labels x n_genes)."""
    d = D[s]; L = d['lab_in'][d['gy'], d['gx']]
    n = int(d['lab_in'].max()) + 1; ng = len(d['meta']['genes'])
    return sp.csr_matrix((np.ones(len(L), np.float32), (L, d['gid'])), shape=(n, ng)), n


def evaluate(name, s, pred_img=None, cell_pred=None):
    """pred_img: (C, H, W) prediction image; or cell_pred: (n_in_labels, C) painted through lab_in."""
    d = D[s]; lab = d['lab']; tc = d['test_col'][None, :] & (lab > 0)
    if pred_img is None:
        cp = cell_pred.astype(np.float32); lin = d['lab_in']
        pred_img = np.stack([cp[:, j][lin] for j in range(cp.shape[1])])  # paint per-cell values through the input segmentation
    n = int(lab.max()) + 1; cnt = np.bincount(lab[tc], minlength=n).astype(np.float64)
    test_cells = np.flatnonzero(cnt >= 10)
    rows = []
    for j, c in enumerate(CH):
        p = pred_img[j][tc].astype(np.float64); y = np.asarray(d['Yc'][j], np.float32)[tc].astype(np.float64); l = lab[tc]
        pm = np.bincount(l, weights=p, minlength=n) / np.maximum(cnt, 1); ym = np.bincount(l, weights=y, minlength=n) / np.maximum(cnt, 1)
        pr, yr = p - pm[l], y - ym[l]
        within = float(np.corrcoef(pr, yr)[0, 1]) if pr.std() > 1e-9 else 0.0
        rows.append(dict(method=name, section=s, channel=c, perturb=PERT, cell_spearman=float(spearmanr(pm[test_cells], ym[test_cells])[0]),
                         pixel_pearson=float(np.corrcoef(p, y)[0, 1]), within_cell=within))
    return rows


rows = []
# ---------------- TxCellNet runs ----------------
R0 = f'{SIM}/rcc/runs'
ENS = [tuple(e.split('=')) for e in os.environ.get('ENS', '').split(';') if e]  # e.g. ens_cr=sim_plain+sim_aug (mean of prediction images)
KEEP = {m for _, mem in ENS for m in mem.split('+')}; PRED = {}
for tag in TAGS:
    c = json.load(open(f'{R0}/{tag}/cfg.json'))
    net = GenNet(CH, {}, c['cfg'], c['train_genes']).to(dev)
    net.load_state_dict(torch.load(f'{R0}/{tag}/model.pt', map_location=dev, weights_only=True), strict=False); net.eval()
    vocab = net.prepare(c['train_genes']); vpos = {g: i for i, g in enumerate(vocab)}
    for s in SECT:
        d = D[s]; H, W = d['H'], d['W']
        gmap = np.array([vpos.get(g, -1) for g in d['meta']['genes']], np.int64); g = gmap[d['gid']]; k = g >= 0
        ty, tx_, tg = (torch.from_numpy(a[k]).to(dev) for a in (d['gy'], d['gx'], g))
        xd = np.asarray(d['dapi'][0], np.float32); fit = ~d['test_col'][None, :].repeat(H, 0) & (d['lab'] > 0)
        X = torch.from_numpy((xd - xd[fit].mean()) / (xd[fit].std() + 1e-4)).to(dev)
        Lt = torch.from_numpy(d['lab_in']).to(dev)
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
        rows += evaluate(f'TxCellNet:{tag}', s, pred_img=pred)
        if tag in KEEP: PRED[(tag, s)] = pred
        log(tag, s, 'done')
for en, mem in ENS:
    for s in SECT:
        rows += evaluate(f'TxCellNet:{en}', s, pred_img=np.mean([PRED[(m, s)].astype(np.float32) for m in mem.split('+')], 0))
        log(en, s, 'done')

# ---------------- cell-level baselines ----------------
genes = D['rcc']['meta']['genes']; assert genes == D['ccrcc']['meta']['genes'] or True


def features(s):
    Xc, n = cell_counts(s)
    tot = np.asarray(Xc.sum(1)).ravel()
    Fe = np.log1p((sp.diags(1e3 / np.maximum(tot, 1)) @ Xc).toarray()).astype(np.float32)
    return Fe, n


UNION = sorted(set(D['rcc']['meta']['genes']) | set(D['ccrcc']['meta']['genes'])); up = {g: i for i, g in enumerate(UNION)}
FE, TR, TG, NB, CEN, NL = {}, {}, {}, {}, {}, {}
for s in SECT:
    d = D[s]; Fe0, n = features(s)
    Fe = np.zeros((n, len(UNION)), np.float32); Fe[:, [up[g] for g in d['meta']['genes']]] = Fe0
    lab_in = d['lab_in']; yy, xx = np.nonzero(lab_in); L = lab_in[yy, xx]
    cnt = np.bincount(L, minlength=n).astype(np.float64)
    cen = np.stack([np.bincount(L, weights=xx, minlength=n), np.bincount(L, weights=yy, minlength=n)], 1) / np.maximum(cnt, 1)[:, None]
    # training targets: per-cell mean of the noisy synthetic image over the INPUT segmentation, train stripes only
    Y = np.load(f'{SIM}/{s}/Y.npy', mmap_mode='r')
    tgt = np.stack([np.bincount(L, weights=np.asarray(Y[j], np.float32)[yy, xx].astype(np.float64), minlength=n) / np.maximum(cnt, 1) for j in range(len(CH))], 1)
    train = (cnt > 0) & ~d['test_col'][np.clip(cen[:, 0].astype(int), 0, d['W'] - 1)]
    train[0] = False
    nb = NearestNeighbors(n_neighbors=13).fit(cen).kneighbors(cen, return_distance=False)[:, 1:]
    FE[s], TR[s], TG[s], NB[s], CEN[s], NL[s] = Fe, train, tgt.astype(np.float32), nb, cen, n
mu_t = np.concatenate([TG[s][TR[s]] for s in SECT]).mean(0); sd_t = np.concatenate([TG[s][TR[s]] for s in SECT]).std(0) + 1e-6


def ridge(Xtr, Ytr, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Z = (Xtr - mu) / sd
    Wt = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Ytr - Ytr.mean(0)))
    return lambda X: ((X - mu) / sd) @ Wt + Ytr.mean(0)


for name, niche in [('ridge', False), ('ridge_niche', True)]:
    def fx(s): return np.c_[FE[s], FE[s][NB[s]].mean(1)] if niche else FE[s]
    f = ridge(np.concatenate([fx(s)[TR[s]] for s in SECT]), np.concatenate([TG[s][TR[s]] for s in SECT]))
    for s in SECT: rows += evaluate(name, s, cell_pred=f(fx(s)))
    log(name, 'done')

model = NicheTrans(source_length=len(UNION), target_length=len(CH), noise_rate=0.2, dropout_rate=0.2).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4); sched = torch.optim.lr_scheduler.StepLR(opt, 4, 0.1)
allF = torch.from_numpy(np.concatenate([FE[s] for s in SECT])).to(dev)
offs = np.cumsum([0] + [NL[s] for s in SECT])[:-1]
allNB = torch.from_numpy(np.concatenate([NB[s] + o for s, o in zip(SECT, offs)])).to(dev)
allY = torch.from_numpy(np.concatenate([(TG[s] - mu_t) / sd_t for s in SECT]).astype(np.float32)).to(dev)
tr_idx = np.concatenate([np.flatnonzero(TR[s]) + o for s, o in zip(SECT, offs)])
for ep in range(8):
    model.train(); perm = np.random.default_rng(ep).permutation(tr_idx)
    for i in range(0, len(perm) - 1023, 1024):
        b = torch.from_numpy(perm[i:i + 1024]).to(dev)
        loss = F.mse_loss(model(allF[b], allF[allNB[b]]), allY[b]); opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
model.eval()
for s, o in zip(SECT, offs):
    out = []
    with torch.no_grad():
        for i in range(0, NL[s], 8192):
            b = torch.arange(o + i, o + min(i + 8192, NL[s]), device=dev); out.append(model(allF[b], allF[allNB[b]]).cpu().numpy())
    rows += evaluate('NicheTrans', s, cell_pred=np.vstack(out) * sd_t + mu_t)
log('NicheTrans done')

df = pd.DataFrame(rows); df.to_csv(f'{OUT}/sim_rows_{PERT}.csv', index=False)
pd.set_option('display.width', 220)
for met in ['cell_spearman', 'pixel_pearson', 'within_cell']:
    log(f'\n{met} ({PERT}, mean over the two sections)\n' + df.groupby(['method', 'channel'])[met].mean().unstack('channel')[CH].round(3).to_string())
