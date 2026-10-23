"""End-to-end smoke test of TxCellNet on small synthetic sections (CPU or GPU, about a minute on a laptop CPU).

Checks
  1  stage 1: molecule splatting (gene field = sum of the gene embeddings at each pixel + log count) matches a direct sum
  2  stage 1: a few optimisation steps lower the loss; gradients reach the gene embeddings and the cell-neighbourhood
     transformer (zero-initialised output layer becomes non-zero)
  3  whole-section inference (windows, context margins, four flips): molecules of genes outside the vocabulary are dropped
     and do not change the prediction
  4  a target panel that lacks part of the vocabulary still runs; panel matching reads only the panel's genes
  5  stage 2: rectified-flow training step, residual sampling, and the cell-mean projection: every cell mean of the refined
     image equals the stage-1 cell mean
  6  cell-level metrics on a monotone pair (Spearman 1)
Prints '[smoke] ALL PASS' and exits with status 0, or lists the failed checks and exits with status 1.
usage: python smoke_test.py
"""
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # also runs before `pip install -e .`
from txcellnet import GenNet, FlowUNet, geometry, refine, sample, predict_case, label_counts, metrics, default_device, amp

torch.manual_seed(0)
RNG = np.random.default_rng(0)
DEV = default_device()
GENES = [f'GENE{i:02d}' for i in range(24)]          # training vocabulary: three programmes of eight genes
PROTS = ['membrane_A', 'nuclear_B', 'cytoplasm_C']  # one output channel per programme
CFG = dict(rep='learned', enc='sum', own=0)


def make_section(rng, H, W, n_cells, genes=GENES, extra_genes=()):
    """Voronoi cells (radius 6.5 px) of three types; each type expresses one gene programme and one protein channel
    (membrane, nucleus or whole cell). Returns a case dict as written by the case preparation scripts, plus the targets."""
    seeds = np.c_[rng.uniform(5, H - 5, n_cells), rng.uniform(5, W - 5, n_cells)]
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2
    dmin = np.sqrt(d2.min(-1))
    lab = np.where(dmin <= 6.5, d2.argmin(-1) + 1, 0).astype(np.int32)
    ctype = rng.integers(0, 3, n_cells)
    all_genes = list(genes) + list(extra_genes)
    gy, gx, gi = [], [], []
    for k in range(1, n_cells + 1):
        pix = np.flatnonzero(lab.ravel() == k)
        if not len(pix): continue
        n = rng.poisson(40)
        prog = rng.random(n) < 0.8
        g = np.where(prog, ctype[k - 1] * 8 + rng.integers(0, 8, n), rng.integers(0, len(all_genes), n))
        p = rng.choice(pix, n)
        gy.append(p // W); gx.append(p % W); gi.append(g)
    gy, gx, gi = (np.concatenate(a).astype(np.int64) for a in (gy, gx, gi))
    dapi = np.log1p(np.exp(-d2 / (2 * 2.5 ** 2)).sum(-1) * 100)
    dapi = ((dapi - dapi.mean()) / dapi.std()).astype(np.float32)
    b = np.zeros_like(lab, bool)
    b[:-1] |= lab[:-1] != lab[1:]; b[1:] |= lab[1:] != lab[:-1]; b[:, :-1] |= lab[:, :-1] != lab[:, 1:]; b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b &= lab > 0
    t = np.r_[-1, ctype][lab]
    Y = np.stack([(t == 0) & b, (t == 1) & (dmin < 3), (t == 2) & (lab > 0)]).astype(np.float32)
    case = dict(dapi=dapi, lab=lab, genes=all_genes, gy=gy, gx=gx, gidx=gi)
    return case, Y


def batch_inputs(cases, vocab):
    """stack sections of equal size into one batch: (x, (flat, gene id, n, h, w), labels)"""
    pos = {g: i for i, g in enumerate(vocab)}
    H, W = cases[0]['lab'].shape
    flat, gid = [], []
    for b, c in enumerate(cases):
        m = np.array([pos.get(g, -1) for g in c['genes']])[c['gidx']]; k = m >= 0
        flat.append(b * H * W + c['gy'][k] * W + c['gx'][k]); gid.append(m[k])
    x = torch.from_numpy(np.stack([c['dapi'] for c in cases]))[:, None].to(DEV)
    lab = torch.from_numpy(np.stack([c['lab'] for c in cases])).to(DEV)
    tx = (torch.from_numpy(np.concatenate(flat)).to(DEV), torch.from_numpy(np.concatenate(gid)).to(DEV), len(cases), H, W)
    return x, tx, lab


def cell_means(img, lab):
    lr = lab.ravel(); n = int(lr.max()) + 1; cnt = np.maximum(np.bincount(lr, minlength=n), 1)
    return np.stack([np.bincount(lr, weights=np.asarray(img[j], np.float64).ravel(), minlength=n) / cnt for j in range(len(img))], 1)[1:]


STATE = {}


def check_splatting():
    net = GenNet(PROTS, {}, CFG, GENES).to(DEV)
    case, _ = make_section(RNG, 32, 40, 8)
    x, tx, lab = batch_inputs([case], net.prepare(GENES))
    with torch.no_grad():
        g = net.encode(*tx)[0].cpu().numpy()
    E = net.emb.weight.detach().cpu().numpy(); flat, gid = tx[0].cpu().numpy(), tx[1].cpu().numpy()
    ref = np.zeros((32 * 40, E.shape[1] + 1)); np.add.at(ref, flat, np.c_[E[gid], np.ones(len(gid))])
    ref = ref.reshape(32, 40, -1).transpose(2, 0, 1); ref[-1] = np.log1p(ref[-1])
    assert g.shape == (E.shape[1] + 1, 32, 40), g.shape
    assert np.allclose(g, ref, atol=1e-5), float(np.abs(g - ref).max())


def check_training():
    net = GenNet(PROTS, {}, CFG, GENES).to(DEV)
    vocab = net.prepare(GENES)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    data = [make_section(RNG, 64, 64, 30) for _ in range(4)]
    losses = []
    for it in range(30):
        cs = [data[(2 * it) % 4], data[(2 * it + 1) % 4]]
        x, tx, lab = batch_inputs([c for c, _ in cs], vocab)
        y = torch.from_numpy(np.stack([Y for _, Y in cs])).to(DEV)
        with amp(DEV):
            yh = net(x, tx, lab)
        loss = (yh.float() - y).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        if it == 0:
            assert net.emb.weight.grad is not None and net.emb.weight.grad.abs().sum() > 0, 'no gradient on the gene embeddings'
            assert net.cell.out.weight.grad.abs().sum() > 0, 'no gradient on the cell-neighbourhood transformer'
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); losses.append(loss.item())
    first, last = np.mean(losses[:5]), np.mean(losses[-5:])
    assert np.isfinite(losses).all(), losses
    assert last < first, f'loss did not decrease: {first:.4f} -> {last:.4f}'
    assert net.cell.out.weight.abs().sum() > 0, 'transformer output layer still zero'
    net.eval(); STATE['net'] = net
    print(f'         stage-1 loss {first:.4f} -> {last:.4f}')


def check_vocabulary():
    net = STATE['net']; vocab = net.prepare(GENES)
    case, _ = make_section(RNG, 96, 88, 45, extra_genes=('NOVEL1', 'NOVEL2', 'NOVEL3'))
    known = np.isin(np.array(case['genes'])[case['gidx']], GENES)
    assert (~known).sum() > 0, 'test section has no out-of-vocabulary molecules'
    csum, kept, total, small = predict_case(net, vocab, case, Tt=64, O=8, image=True, ds=8)
    assert kept == int(known.sum()) and total == len(known), (kept, int(known.sum()), total)
    clean = dict(case, gy=case['gy'][known], gx=case['gx'][known], gidx=case['gidx'][known])
    csum2, kept2, _, _ = predict_case(net, vocab, clean, Tt=64, O=8)
    assert kept2 == kept and np.allclose(csum, csum2, atol=1e-6), 'out-of-vocabulary molecules changed the prediction'
    pc = csum / np.maximum(label_counts(case['lab']), 1)[:, None]
    assert np.isfinite(pc).all() and pc.shape == (int(case['lab'].max()) + 1, len(PROTS)), pc.shape
    assert small.shape == (len(PROTS), 96 // 8, 88 // 8), small.shape
    STATE['case'] = case


def check_panel():
    net = STATE['net']; vocab = net.prepare(GENES)
    panel = GENES[::2]  # the target panel measures half of the vocabulary
    case, _ = make_section(RNG, 72, 72, 30, genes=panel)
    k = case['gidx'] < len(panel)  # programme genes beyond the panel are not measured
    case.update(gy=case['gy'][k], gx=case['gx'][k], gidx=case['gidx'][k])
    csum, kept, total, _ = predict_case(net, vocab, case, Tt=64, O=8)
    assert kept == total and np.isfinite(csum).all(), (kept, total)
    pm = GenNet(PROTS, {}, dict(CFG, panel_genes=GENES[:12]), GENES).to(DEV).eval()  # panel matching
    pv = pm.prepare(GENES)
    assert sum(not g.startswith('__off__') for g in pv) == 12 and len(pv) == len(GENES)
    case2 = STATE['case']
    _, kept2, _, _ = predict_case(pm, pv, case2, Tt=64, O=8)
    assert kept2 == int(np.isin(np.array(case2['genes'])[case2['gidx']], GENES[:12]).sum()), kept2


def check_stage2():
    net, case = STATE['net'], STATE['case']
    H, W = case['lab'].shape; ph, pw = (-H) % 8, (-W) % 8
    vocab = net.prepare(GENES)
    x, tx, lab_t = batch_inputs([case], vocab)
    flat = tx[0] // W * (W + pw) + tx[0] % W  # molecule pixel index on the padded grid
    with torch.no_grad(), amp(DEV):
        mu = net(F.pad(x, (0, pw, 0, ph)), (flat, tx[1], 1, H + ph, W + pw), F.pad(lab_t, (0, pw, 0, ph)))
    mu = mu[0, :, :H, :W].float().cpu().numpy()
    P = len(PROTS); NC = P + 1 + 3
    flow = FlowUNet(P + NC, P).to(DEV)
    cond = torch.from_numpy(np.concatenate([mu, case['dapi'][None], geometry(case['lab'])], 0).astype(np.float32)).to(DEV)
    cond = F.pad(cond, (0, pw, 0, ph))[None]
    r = torch.randn((1, P) + cond.shape[2:], device=DEV)
    opt = torch.optim.AdamW(flow.parameters(), 2e-4)
    for _ in range(2):  # rectified-flow objective: velocity along the straight path from noise to the residual
        t = torch.sigmoid(torch.randn(1, device=DEV)); z0 = torch.randn_like(r); zt = (1 - t) * z0 + t * r
        with amp(DEV):
            v = flow(torch.cat([zt, cond], 1), t)
        loss = (v.float() - (r - z0)).square().mean(); opt.zero_grad(); loss.backward(); opt.step()
        assert torch.isfinite(loss), loss
    flow.eval()
    rsd = np.full(P, 0.5, np.float32)
    res = sample(flow, cond[0], torch.randn((P,) + cond.shape[2:], device=DEV), 0.85, torch.from_numpy(rsd).to(DEV)[None, :, None, None], steps=4)
    assert res.shape == (P,) + cond.shape[2:] and torch.isfinite(res).all()
    inside = np.ones((H, W), bool); inside[:, : W // 4] = False  # part of the section outside the tissue mask
    out = refine(flow, rsd, mu, case['dapi'], case['lab'], inside=inside, noise=0.85, steps=4)
    assert out.shape == mu.shape and np.isfinite(out).all()
    err = np.abs(cell_means(out, case['lab']) - cell_means(mu, case['lab'])).max()
    assert err < 1e-4, f'cell means changed by stage 2: {err:.2e}'
    incell = case['lab'] > 0
    assert np.abs(out - mu)[:, incell].max() > 1e-3, 'stage 2 added no texture'
    assert np.allclose(out[:, ~inside & ~incell], mu[:, ~inside & ~incell]), 'residual not zero outside the tissue mask'
    print(f'         stage-2 cell means preserved (max |difference| {err:.1e})')


def check_metrics():
    rng = np.random.default_rng(1); t = rng.normal(size=500)
    m = metrics(np.exp(t), t)
    assert abs(m['spearman'] - 1) < 1e-12 and 0.99 < m['auroc'] <= 1, m
    assert np.isnan(metrics(np.ones(500), t)['spearman'])


CHECKS = [('molecule splatting', check_splatting), ('stage-1 training', check_training),
          ('out-of-vocabulary genes dropped', check_vocabulary), ('partial panel and panel matching', check_panel),
          ('stage 2 keeps every cell mean', check_stage2), ('cell-level metrics', check_metrics)]

if __name__ == '__main__':
    t0 = time.time(); failed = []
    print(f'[smoke] torch {torch.__version__}, device {DEV}')
    for name, fn in CHECKS:
        t1 = time.time()
        try:
            fn(); print(f'[smoke] ok    {name} ({time.time() - t1:.1f}s)')
        except Exception:
            failed.append(name); print(f'[smoke] FAIL  {name}'); traceback.print_exc()
    if failed:
        print(f'[smoke] {len(failed)} check(s) failed: {", ".join(failed)}'); sys.exit(1)
    print(f'[smoke] ALL PASS ({time.time() - t0:.0f}s)')
