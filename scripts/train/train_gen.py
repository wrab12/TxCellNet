"""Train a TxCellNet stage-1 network (txcellnet.model.GenNet) jointly on both kidney sections (training stripes of RCC +
ccRCC); at the end the held-out stripes are scored (cell-level Pearson against the vendor per-cell protein values).

Config (env): THIN=0|1    molecule thinning per patch (keep probability ~ LogUniform(0.08, 1))
              GDROP=0..1  gene dropout per patch (with prob .5: drop a random fraction U(0, GDROP) of the panel's genes)
              DAUG=0|1    DAPI appearance perturbation (gain/offset, blur, noise)
              PAUG=p      each patch is randomized (thinning + gene dropout w.p. .5 + DAPI perturbation) with prob p,
                          otherwise left clean: PAUG=0 clean network, PAUG=1 randomized network, PAUG=0.5 mixed network
                          (with THIN=1 GDROP=0.5 DAUG=1); unset = the switches above act independently
              PANEL=name  panel matching: train reading only the genes of panel group `name` (results/bench/panels.json)
              SECTIONS=rcc|ccrcc  train on one section only (cross-patient analysis; both are still evaluated)
              ablations: NODAPI=1 (DAPI zeroed), NONICHE=1 (no cell-niche transformer),
                         PCAEMB=1 (gene vectors fixed to PCA loadings of kidney expression instead of learned)
              SEED, EMA=0.999, TTA=1 (flips at the final evaluation)
usage: train_gen.py OUT_TAG [ITERS]     -> data/xenium_protein_{rcc,ccrcc}/pilot_L2/runs/OUT_TAG/{model.pt,cfg.json,metrics.csv,log.txt}
"""
import json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F

from txcellnet.model import GenNet, gauss_kernel

torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
tag = sys.argv[1]; ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
P, BS = 256, 16
cfg = dict(rep='learned', enc='sum', own=0,
           thin=int(os.environ.get('THIN', 0)), gdrop=float(os.environ.get('GDROP', 0)), daug=int(os.environ.get('DAUG', 0)),
           seed=int(os.environ.get('SEED', 1)), iters=ITERS, paug=float(os.environ.get('PAUG', -1)),
           nodapi=int(os.environ.get('NODAPI', 0)), noniche=int(os.environ.get('NONICHE', 0)), pcaemb=int(os.environ.get('PCAEMB', 0)))
PANEL = os.environ.get('PANEL', '')
EMA, TTA = float(os.environ.get('EMA', 0.999)), int(os.environ.get('TTA', 1))
torch.manual_seed(cfg['seed']); dev = 'cuda'
SIMROOT = os.environ.get('SIMROOT', '')  # semi-synthetic benchmark: same inputs, synthetic targets (sim_make.py)
SD = [f'{SIMROOT}/rcc', f'{SIMROOT}/ccrcc'] if SIMROOT else ['data/xenium_protein_rcc/pilot_L2', 'data/xenium_protein_ccrcc/pilot_L2']; BT = ['rcc', 'ccrcc']
outs = [f'{d}/runs/{tag}' for d in SD]
for o in outs: os.makedirs(o, exist_ok=True)


def log(*a):
    msg = time.strftime('%H:%M:%S') + ' ' + ' '.join(map(str, a))
    print(msg, flush=True)
    for o in outs: open(f'{o}/log.txt', 'a').write(msg + '\n')


metas = [json.load(open(f'{d}/meta.json')) for d in SD]
union = sorted(set(metas[0]['genes']) | set(metas[1]['genes']))
prots = metas[0]['prots']; NP = len(prots); OWN = metas[0]['own_genes']
if PANEL:
    cfg.update(panel=PANEL, panel_genes=sorted(set(json.load(open('results/bench/panels.json'))[PANEL]['genes']) & set(union)))
net = GenNet(prots, OWN, cfg, union).to(dev)
if cfg['pcaemb']:  # fixed gene vectors: loadings of the top-64 PCs of standardised log-normalised kidney expression
    pe = np.load('results/bench/pca_gene_emb.npy'); assert pe.shape == tuple(net.emb.weight.shape), pe.shape
    with torch.no_grad(): net.emb.weight.copy_(torch.from_numpy(pe))
    net.emb.weight.requires_grad_(False)
vocab = net.prepare(union)  # genes the model reads
vpos = {g: i for i, g in enumerate(vocab)}
log(f'cfg { {k: v for k, v in cfg.items() if k != "panel_genes"} }; union genes {len(union)}, model vocab {len(vocab)}, '
    f'genes read {sum(not g.startswith("__off__") for g in vocab)}')


class Sample:
    def __init__(s, sd, meta, btag):
        s.sd, s.btag, s.um = sd, btag, meta['um_per_px']
        valid, tissue = np.load(f'{sd}/valid.npy'), np.load(f'{sd}/tissue.npy')
        s.H, s.W = valid.shape
        edges = np.array(json.load(open(f'results/cell_baseline/{btag}_stripes.json'))['x_edges_um']) / s.um
        test_col = (np.digitize(np.arange(s.W), edges) % 2 == 1)[None, :]
        s.fit_mask, s.test_mask = valid & tissue & ~test_col, valid & tissue & test_col
        ii = np.flatnonzero(s.fit_mask.ravel())[np.random.default_rng(0).choice(int(s.fit_mask.sum()), 2_000_000)]
        X = np.load(f'{sd}/X.npy', mmap_mode='r'); a = np.asarray(X[0], np.float32); v = a.ravel()[ii]
        s.Xt = torch.from_numpy((a - v.mean()) / (v.std() + 1e-4)).to(dev).half()[None]
        s.Lt = torch.from_numpy(np.load(f'{sd}/cells.npy')).to(dev)
        s.Mt = torch.from_numpy(s.fit_mask).to(dev)
        s.Ynp = np.load(f'{sd}/Y.npy', mmap_mode='r')
        s.pidx = [meta['prots'].index(p) for p in prots]
        s.Yt = torch.empty((NP, s.H, s.W), dtype=torch.float16, device=dev)
        s.ymu, s.ysd = torch.zeros(NP), torch.ones(NP)
        for i, j in enumerate(s.pidx):
            a = np.asarray(s.Ynp[j], np.float32); v = a.ravel()[ii]
            s.ymu[i], s.ysd[i] = float(v.mean()), float(v.std()) + 1e-4
            s.Yt[i] = torch.from_numpy((a - s.ymu[i].item()) / s.ysd[i].item()).to(dev).half()
        T = np.load(f'{sd}/tx.npz')
        gmap = np.array([vpos.get(g, -1) for g in meta['genes']], np.int64)
        gid = gmap[T['gid'].astype(np.int64)]
        s.tgy = torch.from_numpy(T['gy'].astype(np.int64)).to(dev)
        s.tgx = torch.from_numpy(T['gx'].astype(np.int64)).to(dev)
        s.tgid = torch.from_numpy(gid).to(dev)
        s.offs, s.TB, s.nbx, s.nby = T['offsets'], int(T['B']), int(T['nbx']), int(T['nby'])
        s.panel = torch.tensor(sorted({vpos[g] for g in meta['genes'] if g in vpos}), device=dev)
        cy, cx = np.nonzero(s.fit_mask[P // 2:s.H - P // 2:8, P // 2:s.W - P // 2:8]); s.cy, s.cx = cy * 8, cx * 8

    def tx_local(s, y0, x0, h, w):
        bx0, bx1 = max(x0, 0) // s.TB, min((x0 + w - 1) // s.TB, s.nbx - 1)
        rs = [(s.offs[r * s.nbx + bx0], s.offs[r * s.nbx + bx1 + 1])
              for r in range(max(y0, 0) // s.TB, min((y0 + h - 1) // s.TB, s.nby - 1) + 1)]
        rs = [(a, b) for a, b in rs if b > a]
        if not rs:
            z = torch.zeros(0, dtype=torch.long, device=dev); return z, z, z
        idx = torch.cat([torch.arange(int(a), int(b), device=dev) for a, b in rs])
        ly, lx, g = s.tgy[idx] - y0, s.tgx[idx] - x0, s.tgid[idx]
        k = (ly >= 0) & (ly < h) & (lx >= 0) & (lx < w) & (g >= 0)
        return ly[k], lx[k], g[k]


S = [Sample(d, m, b) for d, m, b in zip(SD, metas, BT)]
w_s = np.array([s.fit_mask.sum() for s in S], np.float64)
SECS = os.environ.get('SECTIONS', 'rcc,ccrcc').split(','); cfg['sections'] = SECS
w_s *= np.array([b in SECS for b in BT], np.float64); w_s /= w_s.sum()
log('loaded; train px', [int(s.fit_mask.sum()) for s in S])
ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA)) if EMA else None
LR = 1e-3
opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=ITERS, pct_start=0.05)
rng = np.random.default_rng(cfg['seed'])
gen = torch.Generator(device=dev); gen.manual_seed(cfg['seed'])
KB = gauss_kernel(1.0).to(dev)


def daug(x):
    if rng.random() < .5: x = x * float(rng.uniform(.7, 1.3)) + float(rng.uniform(-.3, .3))
    if rng.random() < .3:
        k = gauss_kernel(float(rng.uniform(.5, 1.5))).to(dev); r = len(k) // 2
        x = F.conv2d(F.conv2d(x[None], k.view(1, 1, 1, -1), padding=(0, r)), k.view(1, 1, -1, 1), padding=(r, 0))[0]
    if rng.random() < .3: x = x + 0.1 * torch.randn(x.shape, device=dev, generator=gen)
    return x


def batch():
    xs, ys, ms, ls, fl, gs = [], [], [], [], [], []
    for b in range(BS):
        s = S[rng.choice(2, p=w_s)]
        k = int(rng.integers(len(s.cy))); y0, x0 = int(s.cy[k]), int(s.cx[k])
        sl = (slice(y0, y0 + P), slice(x0, x0 + P))
        x, y, m, l = s.Xt[:, sl[0], sl[1]].float(), s.Yt[:, sl[0], sl[1]], s.Mt[sl], s.Lt[sl]
        ly, lx, g = s.tx_local(y0, x0, P, P)
        a = bool(rng.random() < cfg['paug']) if cfg['paug'] >= 0 else None  # mixed regime: augmented or clean patch
        if cfg['thin'] and len(g) and (a if a is not None else rng.random() > .25):
            keep = torch.rand(len(g), device=dev, generator=gen) < float(np.exp(rng.uniform(np.log(0.08), 0)))
            ly, lx, g = ly[keep], lx[keep], g[keep]
        if cfg['gdrop'] > 0 and len(g) and a is not False and rng.random() < .5:
            f = float(rng.uniform(0, cfg['gdrop']))
            drop = torch.zeros(len(vocab), dtype=torch.bool, device=dev)
            drop[s.panel[torch.rand(len(s.panel), device=dev, generator=gen) < f]] = True
            keep = ~drop[g]; ly, lx, g = ly[keep], lx[keep], g[keep]
        if cfg['daug'] and a is not False: x = daug(x)
        if rng.random() < .5: x, y, m, l, lx = x.flip(-1), y.flip(-1), m.flip(-1), l.flip(-1), P - 1 - lx
        if rng.random() < .5: x, y, m, l, ly = x.flip(-2), y.flip(-2), m.flip(-2), l.flip(-2), P - 1 - ly
        r = int(rng.integers(4))
        x, y, m, l = (t.rot90(r, (-2, -1)) for t in (x, y, m, l))
        for _ in range(r): ly, lx = P - 1 - lx, ly
        xs.append(x); ys.append(y); ms.append(m); ls.append(l)
        fl.append(b * P * P + ly * P + lx); gs.append(g)
    return (torch.stack(xs).float(), torch.stack(ys).float(), torch.stack(ms), torch.stack(ls),
            (torch.cat(fl), torch.cat(gs), BS, P, P))


t0 = time.time()
for it in range(1, ITERS + 1):
    x, y, m, l, tx = batch()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        yh = net(x, tx, l)
    loss = ((yh.float() - y).square() * m[:, None]).sum() / (m.sum() * NP + 1)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step(); sched.step()
    if ema is not None: ema.update_parameters(net)
    if it % 2000 == 0: log(f'it {it} loss {loss.item():.4f} {(time.time() - t0) / it:.3f}s/it')
model = ema.module if ema is not None else net
for o in outs:
    torch.save(model.state_dict(), f'{o}/model.pt'); json.dump(dict(cfg=cfg, train_genes=union), open(f'{o}/cfg.json', 'w'))
model.eval(); model.prepare(union)
if SIMROOT:
    log('simulation run: saved model, evaluation is done by sim_eval.py'); sys.exit(0)

# ---- held-out stripes: cell-level Pearson vs the vendor per-cell values (results/cell_baseline/<s>_true_test.parquet) ----
FLIPS = [[], [-1], [-2], [-1, -2]] if TTA else [[]]
Tt, O = 1024, 64
for s, o in zip(S, outs):
    H, W = s.H, s.W
    labr = np.load(f'{s.sd}/cells.npy').ravel(); ids = pd.read_csv(f'{s.sd}/cell_ids.csv')['cell_id'].astype(str).values
    n = len(ids) + 1; cnt = np.bincount(labr, minlength=n).astype(np.float64)
    csum = torch.zeros(n, NP, dtype=torch.float64, device=dev)
    Lflat = s.Lt
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for y0 in range(0, H, Tt - 2 * O):
            for x0 in range(0, W, Tt - 2 * O):
                ya, xa = max(y0 - O, 0), max(x0 - O, 0)
                yb, xb = min(ya + Tt, H), min(xa + Tt, W)
                hh, ww = yb - ya + (-(yb - ya)) % 8, xb - xa + (-(xb - xa)) % 8
                x = F.pad(s.Xt[:, ya:yb, xa:xb][None].float(), (0, ww - (xb - xa), 0, hh - (yb - ya)))
                lab = F.pad(s.Lt[ya:yb, xa:xb][None], (0, ww - (xb - xa), 0, hh - (yb - ya)))
                ly, lx, g = s.tx_local(ya, xa, hh, ww)
                acc = 0
                for fl in FLIPS:
                    fy = hh - 1 - ly if -2 in fl else ly
                    fx = ww - 1 - lx if -1 in fl else lx
                    oi = model(x.flip(fl) if fl else x, (fy * ww + fx, g, 1, hh, ww), lab.flip(fl) if fl else lab)
                    acc = acc + (oi.flip(fl) if fl else oi)
                oo = (acc / len(FLIPS))[0, :, :yb - ya, :xb - xa].float()
                ye, xe = min(y0 + Tt - 2 * O, H), min(x0 + Tt - 2 * O, W)
                oc = oo[:, y0 - ya:y0 - ya + ye - y0, x0 - xa:x0 - xa + xe - x0]
                csum.index_add_(0, Lflat[y0:ye, x0:xe].reshape(-1).long(), oc.reshape(NP, -1).T.double())
    cm_all = (csum.cpu().numpy() / np.maximum(cnt, 1)[:, None])
    true10x = pd.read_parquet(f'results/cell_baseline/{s.btag}_true_test.parquet')
    Lc = pd.Series(np.arange(1, n), index=ids).reindex(true10x.index).values
    okc = ~np.isnan(Lc); Lc = Lc[okc].astype(int); common = true10x.index[okc]
    ok2 = cnt[Lc] > 0; Lc, common = Lc[ok2], common[ok2]
    rows = [dict(protein=p, cell_pearson_unet=float(np.corrcoef(cm_all[Lc, i], true10x.loc[common, p].values)[0, 1])) for i, p in enumerate(prots)]
    df = pd.DataFrame(rows); df.to_csv(f'{o}/metrics.csv', index=False)
    log(s.btag, 'held-out stripes cell-level Pearson median', round(df.cell_pearson_unet.median(), 3), 'mean', round(df.cell_pearson_unet.mean(), 3))
log('done')
