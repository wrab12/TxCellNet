"""Full-grid TxCellNet prediction on both kidney sections (train and held-out stripes), averaged over models and flips.
Output (per section): pilot_L2/runs/OUT_NAME/pred_full.npy, float16 (27, H, W), in the models' units (per-section z of
log1p protein). Input normalisation is identical to train_gen.py (DAPI z-scored with train-stripe statistics).
Used as the mean image of stage 2 (train_refine.py).
usage: pred_kidney_full.py TAG[,TAG...] OUT_NAME
"""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F

from txcellnet.model import GenNet

TAGS, OUTN = sys.argv[1].split(','), sys.argv[2]
dev = 'cuda'
SD = ['data/xenium_protein_rcc/pilot_L2', 'data/xenium_protein_ccrcc/pilot_L2']; BT = ['rcc', 'ccrcc']
R0 = SD[0]
prots = json.load(open(f'{R0}/meta.json'))['prots']; OWN = json.load(open(f'{R0}/meta.json'))['own_genes']; NP = len(prots)
nets = []
for t in TAGS:
    c = json.load(open(f'{R0}/runs/{t}/cfg.json'))
    net = GenNet(prots, OWN, c['cfg'], c['train_genes']).to(dev)
    net.load_state_dict(torch.load(f'{R0}/runs/{t}/model.pt', map_location=dev, weights_only=True), strict=False); net.eval()
    nets.append((net, net.prepare(c['train_genes'])))
FL = [[], [-1], [-2], [-1, -2]]
Tt, O = 1024, 64
for sd, bt in zip(SD, BT):
    meta = json.load(open(f'{sd}/meta.json')); um = meta['um_per_px']
    valid, tissue = np.load(f'{sd}/valid.npy'), np.load(f'{sd}/tissue.npy'); H, W = valid.shape
    edges = np.array(json.load(open(f'results/cell_baseline/{bt}_stripes.json'))['x_edges_um']) / um
    fit = valid & tissue & ~(np.digitize(np.arange(W), edges) % 2 == 1)[None, :]
    ii = np.flatnonzero(fit.ravel())[np.random.default_rng(0).choice(int(fit.sum()), 2_000_000)]
    a = np.asarray(np.load(f'{sd}/X.npy', mmap_mode='r')[0], np.float32); v = a.ravel()[ii]
    Xt = torch.from_numpy((a - v.mean()) / (v.std() + 1e-4)).to(dev).half(); del a
    Lt = torch.from_numpy(np.load(f'{sd}/cells.npy')).to(dev)
    T = np.load(f'{sd}/tx.npz')
    ty, tx_ = torch.from_numpy(T['gy'].astype(np.int64)).to(dev), torch.from_numpy(T['gx'].astype(np.int64)).to(dev)
    gid0 = T['gid'].astype(np.int64)
    gids = []
    for net, vocab in nets:
        vpos = {g: i for i, g in enumerate(vocab)}
        gids.append(torch.from_numpy(np.array([vpos.get(g, -1) for g in meta['genes']], np.int64)[gid0]).to(dev))
    out = np.zeros((NP, H, W), np.float16)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for y0 in range(0, H, Tt - 2 * O):
            for x0 in range(0, W, Tt - 2 * O):
                ya, xa = max(y0 - O, 0), max(x0 - O, 0)
                yb, xb = min(ya + Tt, H), min(xa + Tt, W)
                hh, ww = yb - ya + (-(yb - ya)) % 8, xb - xa + (-(xb - xa)) % 8
                x = F.pad(Xt[ya:yb, xa:xb][None, None].float(), (0, ww - (xb - xa), 0, hh - (yb - ya)))
                lab = F.pad(Lt[ya:yb, xa:xb][None], (0, ww - (xb - xa), 0, hh - (yb - ya)))
                m = (ty >= ya) & (ty < yb) & (tx_ >= xa) & (tx_ < xb)
                acc = 0
                for (net, _), gg in zip(nets, gids):
                    k = m & (gg >= 0); ly, lx, g = ty[k] - ya, tx_[k] - xa, gg[k]
                    for fl in FL:
                        fy = hh - 1 - ly if -2 in fl else ly
                        fx = ww - 1 - lx if -1 in fl else lx
                        oi = net(x.flip(fl) if fl else x, (fy * ww + fx, g, 1, hh, ww), lab.flip(fl) if fl else lab)
                        acc = acc + (oi.flip(fl) if fl else oi)
                o = (acc / (len(FL) * len(nets)))[0, :, :yb - ya, :xb - xa].float()
                ye, xe = min(y0 + Tt - 2 * O, H), min(x0 + Tt - 2 * O, W)
                out[:, y0:ye, x0:xe] = o[:, y0 - ya:y0 - ya + ye - y0, x0 - xa:x0 - xa + xe - x0].cpu().numpy()
    os.makedirs(f'{sd}/runs/{OUTN}', exist_ok=True)
    np.save(f'{sd}/runs/{OUTN}/pred_full.npy', out)
    json.dump(dict(members=TAGS), open(f'{sd}/runs/{OUTN}/members.json', 'w'))
    print(bt, 'saved', out.shape, flush=True)
