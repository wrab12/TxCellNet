"""TxCellNet stage 2: generative refinement (residual rectified flow) trained jointly on both kidney sections.

Stage 1 (TxCellNet mean model / ensemble) gives the conditional-mean image mu (27 channels, per-section z units).
Stage 2 samples the residual r = y - mu with a rectified-flow U-Net conditioned only on inputs available for any
dataset: mu, DAPI and cell geometry from the segmentation (inside-cell mask, cell boundary, distance to boundary).
Sampling: Euler, STEPS steps from noise scaled by NOISE (temperature). Constraint (default on): the sampled residual is
set to 0 outside tissue and its per-cell mean is removed, so every cell keeps the stage-1 mean intensity (cell-level
accuracy unchanged) while the image gains sub-cellular texture.
Training pixels = train stripes of RCC + ccRCC; evaluation = held-out stripes.
Metrics per protein (vs measured image, test stripes): cell_r (vs 10x values), pix_r, within_r (pixel deviation from
its cell mean, pred vs measured), sharp (mean |gradient| ratio), hfpow (power at |f| > .2 cycles/px ratio),
sd (pixel SD ratio), W1 (1-Wasserstein distance of pixel values, z units).
Network, geometry channels and sampler: txcellnet.flow.
usage: [ITERS=30000] [STEPS=20] [NOISES=0.85,1.0] [LOAD=model.pt] train_refine.py MEAN_RUN OUT_TAG
"""
import json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F

from txcellnet.flow import FlowUNet, geometry, sample as flow_sample

torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
mean_run, tag = sys.argv[1:3]
E = os.environ.get
ITERS, STEPS, LOAD = int(E('ITERS', 30000)), int(E('STEPS', 20)), E('LOAD', '')
NOISES = [float(v) for v in E('NOISES', '0.85,1.0').split(',')]
P, BS, dev = 256, 16, 'cuda'
SD = ['data/xenium_protein_rcc/pilot_L2', 'data/xenium_protein_ccrcc/pilot_L2']; BT = ['rcc', 'ccrcc']
outs = [f'{d}/runs/{tag}' for d in SD]
for o in outs: os.makedirs(o, exist_ok=True)
prots = json.load(open(f'{SD[0]}/meta.json'))['prots']; NP = len(prots)


def log(*a):
    msg = time.strftime('%H:%M:%S') + ' ' + ' '.join(map(str, a))
    print(msg, flush=True)
    for o in outs: open(f'{o}/log.txt', 'a').write(msg + '\n')


class Sec:
    def __init__(s, sd, bt):
        meta = json.load(open(f'{sd}/meta.json')); s.sd, s.bt = sd, bt
        valid, tissue = np.load(f'{sd}/valid.npy'), np.load(f'{sd}/tissue.npy'); s.H, s.W = valid.shape
        edges = np.array(json.load(open(f'results/cell_baseline/{bt}_stripes.json'))['x_edges_um']) / meta['um_per_px']
        s.test_col = np.digitize(np.arange(s.W), edges) % 2 == 1
        s.inside = valid & tissue
        s.fit, s.test = s.inside & ~s.test_col[None, :], s.inside & s.test_col[None, :]
        ii = np.flatnonzero(s.fit.ravel())[np.random.default_rng(0).choice(int(s.fit.sum()), 2_000_000)]  # as train_gen.py
        a = np.asarray(np.load(f'{sd}/X.npy', mmap_mode='r')[0], np.float32); v = a.ravel()[ii]
        s.dapi = ((a - v.mean()) / (v.std() + 1e-4)).astype(np.float16)[None]; del a
        s.lab = np.load(f'{sd}/cells.npy'); s.geom = geometry(s.lab)
        Y = np.load(f'{sd}/Y.npy', mmap_mode='r'); pidx = [meta['prots'].index(p) for p in prots]
        s.Y = np.empty((NP, s.H, s.W), np.float16)
        for i, j in enumerate(pidx):  # per-section z of log1p protein, exactly the stage-1 training target
            a = np.asarray(Y[j], np.float32); v = a.ravel()[ii]
            s.Y[i] = ((a - v.mean()) / (v.std() + 1e-4)).astype(np.float16)
        s.mu = np.load(f'{sd}/runs/{mean_run}/pred_full.npy')
        cy, cx = np.nonzero(s.fit[P // 2:s.H - P // 2:8, P // 2:s.W - P // 2:8]); s.cy, s.cx = cy * 8, cx * 8
        s.ii = ii


S = [Sec(d, b) for d, b in zip(SD, BT)]
rsd = np.array([np.concatenate([(s.Y[j].ravel()[s.ii].astype(np.float32) - s.mu[j].ravel()[s.ii].astype(np.float32)) for s in S]).std() + 1e-4
                for j in range(NP)], np.float32)
w_s = np.array([len(s.cy) for s in S], np.float64); w_s /= w_s.sum()
log(f'mean={mean_run}; residual SD per protein (z units) median {np.median(rsd):.3f}; train windows {[len(s.cy) for s in S]}')
NC = NP + 1 + 3  # condition: mu, DAPI, geometry


def cond(s, sl):
    return np.concatenate([s.mu[:, sl[0], sl[1]], s.dapi[:, sl[0], sl[1]], s.geom[:, sl[0], sl[1]]], 0)


rsd_t = torch.from_numpy(rsd).to(dev)[None, :, None, None]
net = FlowUNet(NP + NC, NP).to(dev)
if LOAD:
    net.load_state_dict(torch.load(LOAD, map_location=dev, weights_only=True)); model = net; log('loaded', LOAD)
else:
    ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9995))
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=ITERS, pct_start=0.03)
    rng = np.random.default_rng(1)

    def batch():
        cs, rs, ms = [], [], []
        for _ in range(BS):
            s = S[rng.choice(len(S), p=w_s)]; k = int(rng.integers(len(s.cy)))
            sl = (slice(s.cy[k], s.cy[k] + P), slice(s.cx[k], s.cx[k] + P))
            cs.append(cond(s, sl)); rs.append(s.Y[:, sl[0], sl[1]] - s.mu[:, sl[0], sl[1]]); ms.append(s.fit[sl])
        c, r, m = (torch.from_numpy(np.stack(a)).to(dev, non_blocking=True) for a in (cs, rs, ms))
        c, r = c.float(), r.float() / rsd_t
        if rng.random() < .5: c, r, m = c.flip(-1), r.flip(-1), m.flip(-1)
        if rng.random() < .5: c, r, m = c.flip(-2), r.flip(-2), m.flip(-2)
        q = int(rng.integers(4)); c, r, m = c.rot90(q, (-2, -1)), r.rot90(q, (-2, -1)), m.rot90(q, (-2, -1))
        return c, r, m

    t0 = time.time()
    for it in range(1, ITERS + 1):
        c, r, m = batch()
        t = torch.sigmoid(torch.randn(BS, device=dev))
        z0 = torch.randn_like(r)
        zt = (1 - t)[:, None, None, None] * z0 + t[:, None, None, None] * r
        with torch.autocast('cuda', dtype=torch.bfloat16):
            v = net(torch.cat([zt, c], 1), t)
        loss = ((v.float() - (r - z0)).square() * m[:, None]).sum() / (m.sum() * NP + 1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); ema.update_parameters(net)
        if it % 2000 == 0: log(f'it {it} loss {loss.item():.4f} {(time.time() - t0) / it:.3f}s/it')
    model = ema.module
    for o in outs:
        torch.save(model.state_dict(), f'{o}/model.pt')
        json.dump(dict(mean_run=mean_run, rsd=rsd.tolist(), prots=prots, cond=['mu x27', 'dapi', 'incell', 'boundary', 'dist/6'],
                       steps=STEPS), open(f'{o}/cfg.json', 'w'))
model.eval()


def sample(model, C_, noise, ns, steps=STEPS):
    """C_: (NC, h, w) condition, noise: (NP, h, w) standard normal -> residual in z units (NP, h, w)"""
    return flow_sample(model, C_, noise, ns, rsd_t, steps)


# ---------------- evaluation on held-out stripes ----------------
fy = np.fft.fftfreq(256)[:, None]; fx = np.fft.fftfreq(256)[None, :]; HF = np.sqrt(fy ** 2 + fx ** 2) > 0.2
rows = []
for s, o in zip(S, outs):
    H, W = s.H, s.W
    labr = s.lab.ravel(); n = int(labr.max()) + 1; cnt = np.bincount(labr, minlength=n).astype(np.float64); incell = labr > 0
    ids = pd.read_csv(f'{s.sd}/cell_ids.csv')['cell_id'].astype(str).values
    true10x = pd.read_parquet(f'results/cell_baseline/{s.bt}_true_test.parquet')
    Lc = pd.Series(np.arange(1, len(ids) + 1), index=ids).reindex(true10x.index).values
    okc = ~np.isnan(Lc); Lc = Lc[okc].astype(int); tt = true10x[okc]; ok2 = cnt[Lc] > 0; Lc, tt = Lc[ok2], tt[ok2]
    rng2 = np.random.default_rng(0); pix = rng2.choice(np.flatnonzero(s.test.ravel()), 2_000_000, replace=False)
    pin = pix[incell[pix]]
    crops = []
    while len(crops) < 48:
        y0, x0 = int(rng2.integers(0, H - 256)), int(rng2.integers(0, W - 256))
        if s.test[y0:y0 + 256, x0:x0 + 256].mean() > 0.95: crops.append((y0, x0))

    def stats(img):
        g = np.mean([(np.abs(np.diff(img[y:y + 256, x:x + 256], axis=0))[:, :-1] + np.abs(np.diff(img[y:y + 256, x:x + 256], axis=1))[:-1, :]).mean() for y, x in crops])
        pw = np.mean([(np.abs(np.fft.fft2(img[y:y + 256, x:x + 256] - img[y:y + 256, x:x + 256].mean())) ** 2)[HF].sum() for y, x in crops])
        return g, pw

    REF = {}
    for j, p in enumerate(prots):
        ref = s.Y[j].astype(np.float32); cmr = np.bincount(labr, weights=ref.ravel().astype(np.float64), minlength=n) / np.maximum(cnt, 1)
        REF[j] = (ref.ravel()[pix], np.sort(ref.ravel()[pix]), ref.ravel()[pin] - cmr[labr[pin]], *stats(ref))

    def metrics(img, j, method, cfg):
        rpix, rsort, rwin, gref, pref = REF[j]
        cm = np.bincount(labr, weights=img.ravel().astype(np.float64), minlength=n) / np.maximum(cnt, 1)
        g, pw = stats(img); a = np.sort(img.ravel()[pix])
        return dict(section=s.bt, method=method, config=cfg, protein=prots[j], cell_r=np.corrcoef(cm[Lc], tt[prots[j]].values)[0, 1],
                    pix_r=np.corrcoef(img.ravel()[pix], rpix)[0, 1], within_r=np.corrcoef(img.ravel()[pin] - cm[labr[pin]], rwin)[0, 1],
                    sharp=g / gref, hfpow=pw / pref, sd=a.std() / rsort.std(), W1=float(np.abs(a - rsort).mean()))

    for j in range(NP): rows.append(metrics(s.mu[j].astype(np.float32), j, 'mean (stage 1)', '-'))
    Tt, O = 1024, 64
    tiles = [(y0, x0, min(y0 + Tt - 2 * O, H), min(x0 + Tt - 2 * O, W)) for y0 in range(0, H, Tt - 2 * O)
             for x0 in range(0, W, Tt - 2 * O) if s.test_col[x0:min(x0 + Tt - 2 * O, W)].any()]
    if int(E('MAXTILES', 0)): tiles = tiles[:int(E('MAXTILES'))]  # smoke tests only
    gen = torch.Generator(device=dev)
    for ns in NOISES:
        gen.manual_seed(1000); res = np.zeros((NP, H, W), np.float16)
        for (y0, x0, ye, xe) in tiles:
            ya, xa = max(y0 - O, 0), max(x0 - O, 0); yb, xb = min(ya + Tt, H), min(xa + Tt, W)
            ph, pw_ = (-(yb - ya)) % 8, (-(xb - xa)) % 8
            C_ = F.pad(torch.from_numpy(cond(s, (slice(ya, yb), slice(xa, xb)))).to(dev).float(), (0, pw_, 0, ph))
            nz = torch.randn((NP,) + C_.shape[1:], generator=gen, device=dev)
            rr = sample(model, C_, nz, ns)[:, :yb - ya, :xb - xa]
            res[:, y0:ye, x0:xe] = rr[:, y0 - ya:y0 - ya + ye - y0, x0 - xa:x0 - xa + xe - x0].cpu().numpy()
        ref_img = np.zeros((NP, H, W), np.float16)
        for j in range(NP):
            r = np.where(s.inside.ravel(), res[j].ravel().astype(np.float32), 0)
            cmr = np.bincount(labr, weights=r.astype(np.float64), minlength=n) / np.maximum(cnt, 1)
            img = s.mu[j].astype(np.float32) + (r - np.where(incell, cmr[labr], 0)).reshape(H, W)
            ref_img[j] = img.astype(np.float16)
            rows.append(metrics(img, j, 'refined (constrained)', f'noise{ns}'))
            rows.append(metrics(s.mu[j].astype(np.float32) + res[j].astype(np.float32), j, 'refined (raw)', f'noise{ns}'))
        if ns == NOISES[0]: np.save(f'{o}/refined_noise{ns}.npy', ref_img)  # first temperature only (disk)
        del res, ref_img
        d = pd.DataFrame(rows); d.to_csv(f'{outs[0]}/metrics.csv', index=False); d.to_csv(f'{outs[1]}/metrics.csv', index=False)
        log(s.bt, f'noise {ns}\n' + d[d.section == s.bt].groupby(['method', 'config'])[['cell_r', 'pix_r', 'within_r', 'sharp', 'hfpow', 'sd', 'W1']].median().round(3).to_string())
log('ALL (median over proteins)\n' + pd.DataFrame(rows).groupby(['section', 'method', 'config'])[['cell_r', 'pix_r', 'within_r', 'sharp', 'hfpow', 'sd', 'W1']].median().round(3).to_string())
