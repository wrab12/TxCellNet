"""H&E -> 27-protein baseline with the same backbone as TxCellNet (U-Net + cell-niche transformer + 1x1 head), trained
jointly on the train stripes of both kidney sections; the only difference to TxCellNet is the input (same-section
post-Xenium H&E resampled to the 0.85 um grid instead of transcripts + DAPI).
H&E normalisation: per section, per channel z-score over tissue pixels. Augmentation (COLAUG=1): per-channel gain /
offset jitter in RGB space (stain variation), occasional blur, flips / rotations.
usage: train_he.py OUT_TAG [ITERS]   -> data/xenium_protein_{rcc,ccrcc}/pilot_L2/runs/OUT_TAG/{model.pt,cfg.json,metrics.csv}
"""
import json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

from txcellnet.model import gauss_kernel
from he_model import HENet

torch.backends.cudnn.benchmark = True
tag = sys.argv[1]; ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
P, BS = 256, 16; SEED = int(os.environ.get('SEED', 1)); COLAUG = int(os.environ.get('COLAUG', 1))
torch.manual_seed(SEED); dev = 'cuda'; rng = np.random.default_rng(SEED)
SD = ['data/xenium_protein_rcc/pilot_L2', 'data/xenium_protein_ccrcc/pilot_L2']; BT = ['rcc', 'ccrcc']
HE = ['data/he/kidney_rcc/he_grid.npy', 'data/he/kidney_ccrcc/he_grid.npy']
outs = [f'{d}/runs/{tag}' for d in SD]
for o in outs: os.makedirs(o, exist_ok=True)


def log(*a):
    msg = time.strftime('%H:%M:%S') + ' ' + ' '.join(map(str, a)); print(msg, flush=True)
    for o in outs: open(f'{o}/log.txt', 'a').write(msg + '\n')


metas = [json.load(open(f'{d}/meta.json')) for d in SD]
prots = metas[0]['prots']; NP = len(prots)


class Sample:
    def __init__(s, sd, he, meta, btag):
        s.sd, s.btag = sd, btag
        valid, tissue = np.load(f'{sd}/valid.npy'), np.load(f'{sd}/tissue.npy'); s.H, s.W = valid.shape
        edges = np.array(json.load(open(f'results/cell_baseline/{btag}_stripes.json'))['x_edges_um']) / meta['um_per_px']
        test_col = (np.digitize(np.arange(s.W), edges) % 2 == 1)[None, :]
        s.fit_mask = valid & tissue & ~test_col
        ii = np.flatnonzero(s.fit_mask.ravel())[np.random.default_rng(0).choice(int(s.fit_mask.sum()), 2_000_000)]
        a = np.load(he).astype(np.float32); assert a.shape[1:] == (s.H, s.W), (a.shape, s.H, s.W)
        s.he_raw = torch.from_numpy(a).to(dev).half()
        s.mu = torch.tensor([float(a[c].ravel()[ii].mean()) for c in range(3)], device=dev).view(3, 1, 1)
        s.sd_ = torch.tensor([float(a[c].ravel()[ii].std()) + 1e-4 for c in range(3)], device=dev).view(3, 1, 1)
        s.Lt = torch.from_numpy(np.load(f'{sd}/cells.npy')).to(dev); s.Mt = torch.from_numpy(s.fit_mask).to(dev)
        Y = np.load(f'{sd}/Y.npy', mmap_mode='r'); pidx = [meta['prots'].index(p) for p in prots]
        s.Yt = torch.empty((NP, s.H, s.W), dtype=torch.float16, device=dev)
        for i, j in enumerate(pidx):
            y = np.asarray(Y[j], np.float32); v = y.ravel()[ii]
            s.Yt[i] = torch.from_numpy((y - v.mean()) / (v.std() + 1e-4)).to(dev).half()
        cy, cx = np.nonzero(s.fit_mask[P // 2:s.H - P // 2:8, P // 2:s.W - P // 2:8]); s.cy, s.cx = cy * 8, cx * 8


S = [Sample(d, h, m, b) for d, h, m, b in zip(SD, HE, metas, BT)]
w_s = np.array([s.fit_mask.sum() for s in S], np.float64); w_s /= w_s.sum()
net = HENet(NP).to(dev)
ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=ITERS, pct_start=0.05)
log(f'H&E baseline: iters {ITERS} colaug {COLAUG} seed {SEED}')


def batch():
    xs, ys, ms, ls = [], [], [], []
    for b in range(BS):
        s = S[rng.choice(2, p=w_s)]
        k = int(rng.integers(len(s.cy))); y0, x0 = int(s.cy[k]), int(s.cx[k])
        sl = (slice(y0, y0 + P), slice(x0, x0 + P))
        x = s.he_raw[:, sl[0], sl[1]].float()
        if COLAUG:
            x = x * torch.tensor(rng.uniform(.9, 1.1, 3), device=dev, dtype=torch.float32).view(3, 1, 1) \
                + torch.tensor(rng.uniform(-.05, .05, 3), device=dev, dtype=torch.float32).view(3, 1, 1)
            if rng.random() < .3:
                kk = gauss_kernel(float(rng.uniform(.5, 1.5))).to(dev); r = len(kk) // 2
                x = F.conv2d(F.conv2d(x[:, None], kk.view(1, 1, 1, -1), padding=(0, r)), kk.view(1, 1, -1, 1), padding=(r, 0))[:, 0]
        x = (x - s.mu) / s.sd_
        y, m, l = s.Yt[:, sl[0], sl[1]], s.Mt[sl], s.Lt[sl]
        if rng.random() < .5: x, y, m, l = x.flip(-1), y.flip(-1), m.flip(-1), l.flip(-1)
        if rng.random() < .5: x, y, m, l = x.flip(-2), y.flip(-2), m.flip(-2), l.flip(-2)
        r = int(rng.integers(4)); x, y, m, l = (t.rot90(r, (-2, -1)) for t in (x, y, m, l))
        xs.append(x); ys.append(y); ms.append(m); ls.append(l)
    return torch.stack(xs), torch.stack(ys).float(), torch.stack(ms), torch.stack(ls)


t0 = time.time()
for it in range(1, ITERS + 1):
    x, y, m, l = batch()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        yh = net(x, l)
    loss = ((yh.float() - y).square() * m[:, None]).sum() / (m.sum() * NP + 1)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step(); sched.step(); ema.update_parameters(net)
    if it % 2000 == 0: log(f'it {it} loss {loss.item():.4f} {(time.time() - t0) / it:.3f}s/it')
model = ema.module; model.eval()
for o in outs:
    torch.save(model.state_dict(), f'{o}/model.pt'); json.dump(dict(cfg=dict(input='he', colaug=COLAUG, seed=SEED, iters=ITERS)), open(f'{o}/cfg.json', 'w'))

FL = [[], [-1], [-2], [-1, -2]]; Tt, O = 1024, 64
for s, o in zip(S, outs):
    H, W = s.H, s.W
    labr = np.load(f'{s.sd}/cells.npy').ravel(); ids = pd.read_csv(f'{s.sd}/cell_ids.csv')['cell_id'].astype(str).values
    n = len(ids) + 1; cnt = np.bincount(labr, minlength=n).astype(np.float64)
    csum = torch.zeros(n, NP, dtype=torch.float64, device=dev)
    Xn = (s.he_raw.float() - s.mu) / s.sd_
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for y0 in range(0, H, Tt - 2 * O):
            for x0 in range(0, W, Tt - 2 * O):
                ya, xa = max(y0 - O, 0), max(x0 - O, 0); yb, xb = min(ya + Tt, H), min(xa + Tt, W)
                hh, ww = yb - ya + (-(yb - ya)) % 8, xb - xa + (-(xb - xa)) % 8
                x = F.pad(Xn[:, ya:yb, xa:xb][None], (0, ww - (xb - xa), 0, hh - (yb - ya)))
                lab = F.pad(s.Lt[ya:yb, xa:xb][None], (0, ww - (xb - xa), 0, hh - (yb - ya)))
                acc = sum((model(x.flip(f) if f else x, lab.flip(f) if f else lab).flip(f) if f else model(x, lab)) for f in FL) / len(FL)
                oo = acc[0, :, :yb - ya, :xb - xa].float()
                ye, xe = min(y0 + Tt - 2 * O, H), min(x0 + Tt - 2 * O, W)
                oc = oo[:, y0 - ya:y0 - ya + ye - y0, x0 - xa:x0 - xa + xe - x0]
                csum.index_add_(0, s.Lt[y0:ye, x0:xe].reshape(-1).long(), oc.reshape(NP, -1).T.double())
    cm = csum.cpu().numpy() / np.maximum(cnt, 1)[:, None]
    t10 = pd.read_parquet(f'results/cell_baseline/{s.btag}_true_test.parquet')
    Lc = pd.Series(np.arange(1, n), index=ids).reindex(t10.index).values; ok = ~np.isnan(Lc); Lc = Lc[ok].astype(int); com = t10.index[ok]
    ok2 = cnt[Lc] > 0; Lc, com = Lc[ok2], com[ok2]
    df = pd.DataFrame([dict(protein=p, cell_pearson_unet=float(np.corrcoef(cm[Lc, i], t10.loc[com, p].values)[0, 1])) for i, p in enumerate(prots)])
    df.to_csv(f'{o}/metrics.csv', index=False)
    log(s.btag, 'held-out stripes cell-level Pearson median', round(df.cell_pearson_unet.median(), 3), 'mean', round(df.cell_pearson_unet.mean(), 3))
log('done')
