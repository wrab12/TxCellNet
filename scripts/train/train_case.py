"""Within-dataset TxCellNet: train on the training stripes of ONE prepared case, reading all of its panel genes and
predicting its own measured channels (Y.npy from prep_targets.py), then score the held-out stripes at the cell level
with the same cells, markers and metric as the benchmark (Spearman of per-cell means vs the measured per-cell values).

Env: PAUG (default 0 = clean), THIN / GDROP / DAUG (randomization, used when PAUG > 0), SEED, EMA=0.999
usage: train_case.py CASE TAG [ITERS=12000]
   -> data/cases/CASE/runs/TAG/{model.pt, cfg.json, cellpred.npy}, results/bench/within/CASE__TAG.csv
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, torch, torch.nn.functional as F
from scipy.stats import spearmanr

from txcellnet.model import GenNet, gauss_kernel

torch.backends.cudnn.benchmark = True; torch.set_num_threads(8)
case, tag = sys.argv[1:3]; ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
E = os.environ.get
cfg = dict(rep='learned', enc='sum', own=0, thin=int(E('THIN', 1)), gdrop=float(E('GDROP', 0.5)), daug=int(E('DAUG', 1)),
           paug=float(E('PAUG', 0)), seed=int(E('SEED', 1)), iters=ITERS)
P, BS, UM, dev = 256, 16, 0.85, 'cuda'
ACC = int(E('ACCUM', 1)); MB = BS // ACC  # gradient accumulation for dense 5K sections: ACC micro-batches of BS/ACC patches = one batch of BS
torch.manual_seed(cfg['seed']); rng = np.random.default_rng(cfg['seed'])
cd = f'data/cases/{case}'; out = f'{cd}/runs/{tag}'; os.makedirs(out, exist_ok=True); os.makedirs('results/bench/within', exist_ok=True)


def log(*a):
    msg = time.strftime('%H:%M:%S') + ' ' + ' '.join(map(str, a)); print(msg, flush=True); open(f'{out}/log.txt', 'a').write(msg + '\n')


meta = json.load(open(f'{cd}/meta.json')); yj = json.load(open(f'{cd}/Y.json')); chans = yj['channels']; NP = len(chans)
Y = np.load(f'{cd}/Y.npy'); ymask = np.load(f'{cd}/ymask.npy'); lab = np.load(f'{cd}/lab.npy'); H, W = lab.shape
edges = np.array(json.load(open(f'{cd}/split.json'))['x_edges_px'])
test_col = np.digitize(np.arange(W), edges) % 2 == 1
fit = ymask & ~test_col[None, :] & (lab >= 0)
allg = os.path.exists(f'{cd}/tx_all.npz')
T = np.load(f'{cd}/tx_all.npz' if allg else f'{cd}/tx.npz'); genes = json.load(open(f'{cd}/genes_all.json' if allg else f'{cd}/genes.json'))
gy, gx, gi = T['gy'].astype(np.int64), T['gx'].astype(np.int64), T['gidx'].astype(np.int64)
# block index for fast patch queries
B = 64; nby, nbx = (H + B - 1) // B, (W + B - 1) // B
bid = (gy // B) * nbx + gx // B; order = np.argsort(bid, kind='stable'); gy, gx, gi, bid = gy[order], gx[order], gi[order], bid[order]
offs = np.searchsorted(bid, np.arange(nby * nbx + 1))
tgy, tgx, tgi = (torch.from_numpy(a).to(dev) for a in (gy, gx, gi))
log(f'{case}: grid {H}x{W}, channels {chans}, genes {len(genes)} ({"all panel" if allg else "vocabulary"}), molecules {len(gi)}, '
    f'train px {int(fit.sum())}, cfg {cfg}')

net = GenNet(chans, {}, cfg, genes).to(dev); net.prepare(genes)
Xt = torch.from_numpy(np.load(f'{cd}/dapi.npy').astype(np.float16)).to(dev)[None]
Yt = torch.from_numpy(Y).to(dev); Mt = torch.from_numpy(fit).to(dev); Lt = torch.from_numpy(lab).to(dev)
panel = torch.arange(len(genes), device=dev)
cy, cx = np.nonzero(fit[P // 2:H - P // 2:8, P // 2:W - P // 2:8]); cy, cx = cy * 8, cx * 8
assert len(cy), 'no training windows'


def tx_local(y0, x0, h, w):
    rs = [(offs[r * nbx + max(x0, 0) // B], offs[r * nbx + min((x0 + w - 1) // B, nbx - 1) + 1]) for r in range(max(y0, 0) // B, min((y0 + h - 1) // B, nby - 1) + 1)]
    rs = [(a, b) for a, b in rs if b > a]
    if not rs: z = torch.zeros(0, dtype=torch.long, device=dev); return z, z, z
    idx = torch.cat([torch.arange(int(a), int(b), device=dev) for a, b in rs])
    ly, lx, g = tgy[idx] - y0, tgx[idx] - x0, tgi[idx]
    k = (ly >= 0) & (ly < h) & (lx >= 0) & (lx < w)
    return ly[k], lx[k], g[k]


gen = torch.Generator(device=dev); gen.manual_seed(cfg['seed'])


def daug(x):
    if rng.random() < .5: x = x * float(rng.uniform(.7, 1.3)) + float(rng.uniform(-.3, .3))
    if rng.random() < .3:
        k = gauss_kernel(float(rng.uniform(.5, 1.5))).to(dev); r = len(k) // 2
        x = F.conv2d(F.conv2d(x[None], k.view(1, 1, 1, -1), padding=(0, r)), k.view(1, 1, -1, 1), padding=(r, 0))[0]
    if rng.random() < .3: x = x + 0.1 * torch.randn(x.shape, device=dev, generator=gen)
    return x


def batch(nb=BS):
    xs, ys, ms, ls, fl, gs = [], [], [], [], [], []
    for b in range(nb):
        k = int(rng.integers(len(cy))); y0, x0 = int(cy[k]), int(cx[k]); sl = (slice(y0, y0 + P), slice(x0, x0 + P))
        x, y, m, l = Xt[:, sl[0], sl[1]].float(), Yt[:, sl[0], sl[1]].float(), Mt[sl], Lt[sl]
        ly, lx, g = tx_local(y0, x0, P, P)
        a = bool(rng.random() < cfg['paug'])
        if a and len(g):
            keep = torch.rand(len(g), device=dev, generator=gen) < float(np.exp(rng.uniform(np.log(0.08), 0))); ly, lx, g = ly[keep], lx[keep], g[keep]
            if rng.random() < .5 and len(g):
                drop = torch.zeros(len(genes), dtype=torch.bool, device=dev); drop[panel[torch.rand(len(panel), device=dev, generator=gen) < float(rng.uniform(0, cfg['gdrop']))]] = True
                keep = ~drop[g]; ly, lx, g = ly[keep], lx[keep], g[keep]
            x = daug(x)
        if rng.random() < .5: x, y, m, l, lx = x.flip(-1), y.flip(-1), m.flip(-1), l.flip(-1), P - 1 - lx
        if rng.random() < .5: x, y, m, l, ly = x.flip(-2), y.flip(-2), m.flip(-2), l.flip(-2), P - 1 - ly
        r = int(rng.integers(4)); x, y, m, l = (t.rot90(r, (-2, -1)) for t in (x, y, m, l))
        for _ in range(r): ly, lx = P - 1 - lx, ly
        xs.append(x); ys.append(y); ms.append(m); ls.append(l); fl.append(b * P * P + ly * P + lx); gs.append(g)
    return torch.stack(xs), torch.stack(ys), torch.stack(ms), torch.stack(ls), (torch.cat(fl), torch.cat(gs), nb, P, P)


ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(float(E('EMA', 0.999))))
opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=ITERS, pct_start=0.05)
t0 = time.time()
for it in range(1, ITERS + 1):
    mbs = [batch(MB) for _ in range(ACC)]  # same patches, in the same order, as one batch of BS
    D = sum(mb[2].sum() for mb in mbs) * NP + 1; tot = 0.0
    opt.zero_grad(set_to_none=True)
    for x, y, m, l, tx in mbs:
        with torch.autocast('cuda', dtype=torch.bfloat16): yh = net(x, tx, l)
        loss = ((yh.float() - y).square() * m[:, None]).sum() / D; loss.backward(); tot = tot + loss.detach()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); ema.update_parameters(net)
    if it % 2000 == 0: log(f'it {it} loss {tot.item():.4f} {(time.time() - t0) / it:.3f}s/it')
model = ema.module; model.eval()
torch.save(model.state_dict(), f'{out}/model.pt'); json.dump(dict(cfg=cfg, train_genes=genes, channels=chans), open(f'{out}/cfg.json', 'w'))

# ---- whole-grid prediction (1024 tiles, 64 px margins, 4 flips) -> per-cell means ----
n = int(lab.max()) + 1; csum = torch.zeros(n, NP, dtype=torch.float64, device=dev); Tt, O = 1024, 64
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for y0 in range(0, H, Tt - 2 * O):
        for x0 in range(0, W, Tt - 2 * O):
            ya, xa = max(y0 - O, 0), max(x0 - O, 0); yb, xb = min(ya + Tt, H), min(xa + Tt, W)
            hh, ww = yb - ya + (-(yb - ya)) % 8, xb - xa + (-(xb - xa)) % 8
            x = F.pad(Xt[:, ya:yb, xa:xb][None].float(), (0, ww - (xb - xa), 0, hh - (yb - ya)))
            lb = F.pad(Lt[ya:yb, xa:xb][None], (0, ww - (xb - xa), 0, hh - (yb - ya)))
            ly, lx, g = tx_local(ya, xa, hh, ww); acc = 0
            for fl in ([], [-1], [-2], [-1, -2]):
                fy = hh - 1 - ly if -2 in fl else ly; fx = ww - 1 - lx if -1 in fl else lx
                oi = model(x.flip(fl) if fl else x, (fy * ww + fx, g, 1, hh, ww), lb.flip(fl) if fl else lb); acc = acc + (oi.flip(fl) if fl else oi)
            o = (acc / 4)[0, :, :yb - ya, :xb - xa].float(); ye, xe = min(y0 + Tt - 2 * O, H), min(x0 + Tt - 2 * O, W)
            csum.index_add_(0, Lt[y0:ye, x0:xe].reshape(-1).long(), o[:, y0 - ya:y0 - ya + ye - y0, x0 - xa:x0 - xa + xe - x0].reshape(NP, -1).T.double())
cnt = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
cm = csum.cpu().numpy() / np.maximum(cnt, 1)[:, None]
cells = pd.read_parquet(f'{cd}/cells.parquet'); labels = cells['label'].values
pc = cm[labels]; np.save(f'{out}/cellpred.npy', pc.astype(np.float32))

# ---- score held-out stripes ----
Xc = sp.load_npz(f'{cd}/counts.npz').tocsr()
ok = (cells['area_px'].values >= 10) & (np.asarray(Xc.sum(1)).ravel() >= 1)
if 'eval_mask' in meta: ok &= cells[meta['eval_mask']].values.astype(bool)
ys_, xs_ = np.nonzero(lab); lv = lab[ys_, xs_]; xmean = np.bincount(lv, weights=xs_, minlength=n) / np.maximum(np.bincount(lv, minlength=n), 1)  # cell x (px)
te = ok & (np.digitize(xmean[labels], edges) % 2 == 1)
rows = []
for key, mk in meta['markers'].items():
    if key not in yj['markers']: continue
    t = cells[mk['truth']].values.astype(np.float64)
    if mk.get('log', True): t = np.log1p(np.clip(t - mk.get('offset', 0), 0, None))
    p = pc[:, chans.index(yj['markers'][key])]
    rows.append(dict(case=case, marker=key, method=f'within:{tag}', spearman=float(spearmanr(p[te], t[te])[0]),
                     pearson=float(np.corrcoef(p[te], t[te])[0, 1]), n=int(te.sum())))
df = pd.DataFrame(rows); df.to_csv(f'results/bench/within/{case}__{tag}.csv', index=False)
log(case, 'held-out stripes: median Spearman', round(df.spearman.median(), 3), 'n cells', int(te.sum()))
