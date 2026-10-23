"""Whole-section inference of a stage-1 network: windows of up to 1024 px with 64-px context margins (only the 896-px
core of each window is kept), four flips averaged, per-cell sums accumulated window by window."""
import numpy as np
import torch
import torch.nn.functional as F

from .utils import amp

FLIPS = [[], [-1], [-2], [-1, -2]]


def predict_case(net, vocab, case, thin=1.0, Tt=1024, O=64, flips=True, seed=0, image=False, ds=8, device=None):
    """net: GenNet; vocab: net.prepare(...) (genes the model reads, order = gene ids); case: dict with dapi, lab, genes, gy,
    gx, gidx (txcellnet.io.load_case_image). Molecules of genes outside `vocab` are not read.
    Returns (cell_sum (n_labels, P) float64 incl. label 0, molecules read, molecules total, optional ds-x downsampled image).
    Per-cell means: cell_sum / txcellnet.io.label_counts(case['lab'])[:, None]."""
    device = torch.device(device) if device is not None else next(net.parameters()).device
    NP = net.head.out_channels
    H, W = case['lab'].shape
    gpos = {g: i for i, g in enumerate(vocab)}
    cmap = np.array([gpos.get(g, -1) for g in case['genes']], np.int64)
    gid = cmap[case['gidx']]; k = gid >= 0
    if thin < 1:
        k &= np.random.default_rng(seed).random(len(k)) < thin
    ty, tx_, tg = (torch.from_numpy(np.asarray(a)[k]).to(device) for a in (case['gy'], case['gx'], gid))
    X = torch.from_numpy(case['dapi']).to(device); Lt = torch.from_numpy(case['lab']).to(device)
    FL = FLIPS if flips else [[]]
    nl = int(case['lab'].max()) + 1
    csum = torch.zeros(nl, NP, dtype=torch.float64, device=device)
    small = np.zeros((NP, H // ds, W // ds), np.float16) if image else None
    with torch.no_grad(), amp(device):
        for y0 in range(0, H, Tt - 2 * O):
            for x0 in range(0, W, Tt - 2 * O):
                ya, xa = max(y0 - O, 0), max(x0 - O, 0)
                yb, xb = min(ya + Tt, H), min(xa + Tt, W)
                hh, ww = yb - ya + (-(yb - ya)) % 8, xb - xa + (-(xb - xa)) % 8
                x = F.pad(X[ya:yb, xa:xb][None, None], (0, ww - (xb - xa), 0, hh - (yb - ya)))
                lab = F.pad(Lt[ya:yb, xa:xb][None], (0, ww - (xb - xa), 0, hh - (yb - ya)))
                m = (ty >= ya) & (ty < yb) & (tx_ >= xa) & (tx_ < xb)
                ly, lx, g = ty[m] - ya, tx_[m] - xa, tg[m]
                o = 0
                for fl in FL:
                    fy = hh - 1 - ly if -2 in fl else ly
                    fx = ww - 1 - lx if -1 in fl else lx
                    oi = net(x.flip(fl) if fl else x, (fy * ww + fx, g, 1, hh, ww), lab.flip(fl) if fl else lab)
                    o = o + (oi.flip(fl) if fl else oi)
                o = (o / len(FL))[0, :, :yb - ya, :xb - xa].float()
                yc, xc = y0 - ya, x0 - xa
                ye, xe = min(y0 + Tt - 2 * O, H), min(x0 + Tt - 2 * O, W)
                oc = o[:, yc:yc + ye - y0, xc:xc + xe - x0]
                lc = Lt[y0:ye, x0:xe].reshape(-1).long()
                csum.index_add_(0, lc, oc.reshape(NP, -1).T.double())
                if image:
                    sy, sx = (ye - y0) // ds, (xe - x0) // ds
                    if sy and sx:
                        blk = oc[:, :sy * ds, :sx * ds].reshape(NP, sy, ds, sx, ds).mean((2, 4))
                        small[:, y0 // ds:y0 // ds + sy, x0 // ds:x0 // ds + sx] = blk.cpu().numpy()
    return csum.cpu().numpy(), int(k.sum()), len(k), small
