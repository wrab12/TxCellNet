"""Cross-section tier, TxCellNet: the within-dataset network of case A (train_case.py, one clean network trained on A's
training stripes reading all of A's panel genes; data/cases/A/runs/TAG) applied unchanged to cases B of the same panel
group (another section of the same dataset). B's molecules are mapped to A's genes (genes A does not have are not read);
whole-grid prediction with 1,024-pixel tiles, 64-pixel margins and 4 flips, per-cell means, exactly as train_case.py.
Scored on B's held-out stripes (the cells of its within run) and on all its evaluated cells (txcellnet.bench.write_xsec).
Env TAG: within_clean (default; method txcellnet) or within_rand (the randomized network of A, train_case.py PAUG=1;
method txcellnet_rand, plus txcellnet_cr = mean of the clean and randomized predictions, the paper's clean + randomized
recipe for transfer, when the clean predictions of the pair exist).
usage: xsec_txcellnet.py A B[,B...]   -> results/bench/xsec/A__B__txcellnet[_rand|_cr].csv, preds/
"""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F

from txcellnet.bench import xsec_test, write_xsec, log
from txcellnet.model import GenNet

torch.backends.cudnn.benchmark = True; torch.set_num_threads(8)
dev = 'cuda'; a, bs = sys.argv[1], sys.argv[2].split(','); TAG = os.environ.get('TAG', 'within_clean')
rd = f'data/cases/{a}/runs/{TAG}'; c = json.load(open(f'{rd}/cfg.json')); chans, genesA = c['channels'], c['train_genes']
net = GenNet(chans, {}, c['cfg'], genesA).to(dev); vocab = net.prepare(genesA)
net.load_state_dict(torch.load(f'{rd}/model.pt', map_location=dev, weights_only=True)); net.eval()
yjA = json.load(open(f'data/cases/{a}/Y.json')); metaA = json.load(open(f'data/cases/{a}/meta.json'))
kA = [k for k in metaA['markers'] if k in yjA['markers']]  # = within_split(A) keys
vpos = {g: i for i, g in enumerate(vocab)}
log(a, TAG, 'channels', chans, 'genes', len(genesA))

for b in bs:
    cd = f'data/cases/{b}'; lab = np.load(f'{cd}/lab.npy'); H, W = lab.shape; NP = len(chans)
    allg = os.path.exists(f'{cd}/tx_all.npz')
    T = np.load(f'{cd}/tx_all.npz' if allg else f'{cd}/tx.npz'); gB = json.load(open(f'{cd}/genes_all.json' if allg else f'{cd}/genes.json'))
    gmap = np.array([vpos.get(g, -1) for g in gB], np.int64)
    gy, gx, gi = T['gy'].astype(np.int64), T['gx'].astype(np.int64), gmap[T['gidx'].astype(np.int64)]
    k = gi >= 0; gy, gx, gi = gy[k], gx[k], gi[k]
    log(b, 'molecules read', int(k.sum()), 'of', len(k), '| genes of B read', int((gmap >= 0).sum()), 'of', len(gB))
    B_ = 64; nby, nbx = (H + B_ - 1) // B_, (W + B_ - 1) // B_
    bid = (gy // B_) * nbx + gx // B_; order = np.argsort(bid, kind='stable'); gy, gx, gi, bid = gy[order], gx[order], gi[order], bid[order]
    offs = np.searchsorted(bid, np.arange(nby * nbx + 1))
    tgy, tgx, tgi = (torch.from_numpy(x_).to(dev) for x_ in (gy, gx, gi))

    def tx_local(y0, x0, h, w):  # train_case.py
        rs = [(offs[r * nbx + max(x0, 0) // B_], offs[r * nbx + min((x0 + w - 1) // B_, nbx - 1) + 1]) for r in range(max(y0, 0) // B_, min((y0 + h - 1) // B_, nby - 1) + 1)]
        rs = [(p, q) for p, q in rs if q > p]
        if not rs: z = torch.zeros(0, dtype=torch.long, device=dev); return z, z, z
        idx = torch.cat([torch.arange(int(p), int(q), device=dev) for p, q in rs])
        ly, lx, g = tgy[idx] - y0, tgx[idx] - x0, tgi[idx]
        m = (ly >= 0) & (ly < h) & (lx >= 0) & (lx < w)
        return ly[m], lx[m], g[m]

    Xt = torch.from_numpy(np.load(f'{cd}/dapi.npy').astype(np.float16)).to(dev)[None]; Lt = torch.from_numpy(lab).to(dev)
    n = int(lab.max()) + 1; csum = torch.zeros(n, NP, dtype=torch.float64, device=dev); Tt, O = 1024, 64
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):  # whole-grid prediction (train_case.py)
        for y0 in range(0, H, Tt - 2 * O):
            for x0 in range(0, W, Tt - 2 * O):
                ya, xa = max(y0 - O, 0), max(x0 - O, 0); yb, xb = min(ya + Tt, H), min(xa + Tt, W)
                hh, ww = yb - ya + (-(yb - ya)) % 8, xb - xa + (-(xb - xa)) % 8
                x = F.pad(Xt[:, ya:yb, xa:xb][None].float(), (0, ww - (xb - xa), 0, hh - (yb - ya)))
                lb = F.pad(Lt[ya:yb, xa:xb][None], (0, ww - (xb - xa), 0, hh - (yb - ya)))
                ly, lx, g = tx_local(ya, xa, hh, ww); acc = 0
                for fl in ([], [-1], [-2], [-1, -2]):
                    fy = hh - 1 - ly if -2 in fl else ly; fx = ww - 1 - lx if -1 in fl else lx
                    oi = net(x.flip(fl) if fl else x, (fy * ww + fx, g, 1, hh, ww), lb.flip(fl) if fl else lb); acc = acc + (oi.flip(fl) if fl else oi)
                o = (acc / 4)[0, :, :yb - ya, :xb - xa].float(); ye, xe = min(y0 + Tt - 2 * O, H), min(x0 + Tt - 2 * O, W)
                csum.index_add_(0, Lt[y0:ye, x0:xe].reshape(-1).long(), o[:, y0 - ya:y0 - ya + ye - y0, x0 - xa:x0 - xa + xe - x0].reshape(NP, -1).T.double())
    cnt = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
    cm = csum.cpu().numpy() / np.maximum(cnt, 1)[:, None]
    B, keys, ia, teB, okB, TB = xsec_test(b, kA)
    pc = cm[B['cells']['label'].values][:, [chans.index(yjA['markers'][k]) for k in kA]]
    pred = np.full((len(B['cells']), len(keys)), np.nan, np.float32); pred[okB] = pc[okB][:, ia]
    METHOD = {'within_clean': 'txcellnet', 'within_rand': 'txcellnet_rand'}.get(TAG, f'txcellnet_{TAG}')
    write_xsec(a, b, METHOD, keys, pred, TB, teB, okB)
    fc = f'results/bench/xsec/preds/{a}__{b}__txcellnet.npy'
    if TAG == 'within_rand' and os.path.exists(fc):  # clean + randomized (both in the z-units of A's training targets)
        write_xsec(a, b, 'txcellnet_cr', keys, (np.load(fc).astype(np.float32) + pred) / 2, TB, teB, okB)
    del Xt, Lt, tgy, tgx, tgi, csum; torch.cuda.empty_cache()
log('done')
