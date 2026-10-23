"""TxCellNet stage 2: generative refinement with a residual rectified flow that keeps every cell mean.

A time-conditioned U-Net (FiLM blocks) learns the velocity that carries standard normal noise to the normalised residual
r = (Y - mu) / sigma_r, conditioned on inputs available for any section: the stage-1 mean mu, DAPI and three cell
geometry channels (inside-cell mask, cell boundary, distance to the boundary). Sampling: Euler steps from scaled noise;
the sampled residual is set to zero outside tissue and its per-cell mean is removed, so each cell keeps its stage-1 mean.
"""
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import amp, default_device


class TBlock(nn.Module):
    def __init__(s, i, o, td):
        super().__init__()
        s.c1, s.n1 = nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o)
        s.c2, s.n2 = nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o)
        s.film = nn.Linear(td, 2 * o); nn.init.zeros_(s.film.weight); nn.init.zeros_(s.film.bias)
        s.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(s, x, te):
        sc, sh = s.film(te)[:, :, None, None].chunk(2, 1)
        h = F.silu(s.n1(s.c1(x)) * (1 + sc) + sh)
        return F.silu(s.n2(s.c2(h))) + s.skip(x)


class FlowUNet(nn.Module):
    def __init__(s, cin, cout, w=(64, 128, 256, 512), td=256):
        super().__init__()
        s.td = td
        s.tmlp = nn.Sequential(nn.Linear(td, td), nn.SiLU(), nn.Linear(td, td))
        n = len(w) - 1
        s.inp = TBlock(cin, w[0], td)
        s.down = nn.ModuleList([TBlock(w[i], w[i + 1], td) for i in range(n)])
        s.mid = TBlock(w[-1], w[-1], td)
        s.up = nn.ModuleList([nn.ConvTranspose2d(w[i + 1], w[i], 2, stride=2) for i in reversed(range(n))])
        s.dec = nn.ModuleList([TBlock(2 * w[i], w[i], td) for i in reversed(range(n))])
        s.out = nn.Conv2d(w[0], cout, 1)

    def forward(s, x, t):
        half = s.td // 2
        f = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        a = (t * 1000)[:, None] * f[None]
        te = s.tmlp(torch.cat([a.sin(), a.cos()], 1))
        hs = [s.inp(x, te)]
        for d in s.down: hs.append(d(F.max_pool2d(hs[-1], 2), te))
        h = s.mid(hs.pop(), te)
        for u, d in zip(s.up, s.dec): h = d(torch.cat([u(h), hs.pop()], 1), te)
        return s.out(h)


def geometry(lab):
    """inside-cell mask, cell-boundary pixels, distance to the boundary (px / 6, capped at 1) -> (3, H, W) float16"""
    incell = lab > 0
    b = np.zeros_like(incell)
    b[:-1] |= lab[:-1] != lab[1:]; b[1:] |= lab[1:] != lab[:-1]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]; b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b &= incell
    dist = cv2.distanceTransform((incell & ~b).astype(np.uint8), cv2.DIST_L2, 3)
    return np.stack([incell, b, np.minimum(dist / 6, 1)]).astype(np.float16)


@torch.no_grad()
def sample(model, C_, noise, ns, rsd_t, steps=20):
    """C_: (NC, h, w) condition, noise: (P, h, w) standard normal, ns: noise scale (temperature), rsd_t: (1, P, 1, 1)
    residual SDs -> sampled residual (P, h, w) in the units of the stage-1 mean (Euler integration from t=0 to t=1)."""
    z = noise[None] * ns; c = C_[None]
    with amp(c.device):
        for k in range(steps):
            t = torch.full((1,), k / steps, device=c.device)
            z = z + model(torch.cat([z, c], 1), t).float() / steps
    return (z * rsd_t)[0]


def remove_cell_mean(r, lab):
    """r: (P, H, W) residual, lab: (H, W) cell labels -> r minus its per-cell mean on cell pixels (background unchanged)."""
    P, h, w = r.shape
    lr = lab.ravel(); n = int(lr.max()) + 1; cnt = np.maximum(np.bincount(lr, minlength=n), 1)
    out = np.array(r, np.float64)
    for j in range(P):
        cm = np.bincount(lr, weights=out[j].ravel(), minlength=n) / cnt
        out[j] -= np.where(lr > 0, cm[lr], 0).reshape(h, w)
    return out


def load_refiner(run_dir, device=None):
    """stage-2 network of a train_refine.py run -> (net, residual SDs, number of Euler steps)"""
    device = torch.device(device) if device is not None else default_device()
    c = json.load(open(f'{run_dir}/cfg.json')); NP = len(c['prots'])
    net = FlowUNet(NP + NP + 1 + 3, NP).to(device)
    net.load_state_dict(torch.load(f'{run_dir}/model.pt', map_location=device, weights_only=True)); net.eval()
    return net, np.array(c['rsd'], np.float32), c.get('steps', 20)


@torch.no_grad()
def refine(net, rsd, mu, dapi, lab, inside=None, noise=0.85, steps=20, seed=1000, device=None):
    """mu (P, h, w) stage-1 mean (z units), dapi (h, w) z-scored log DAPI, lab (h, w) cell labels -> refined (P, h, w):
    residual sampled by Euler integration of the flow, set to 0 outside `inside` (default: all pixels) and with its
    per-cell mean removed, so each cell keeps its stage-1 mean."""
    device = torch.device(device) if device is not None else next(net.parameters()).device
    P, h, w = mu.shape; ph, pw = (-h) % 8, (-w) % 8
    c = np.concatenate([mu, dapi[None], geometry(lab)], 0).astype(np.float32)
    C_ = F.pad(torch.from_numpy(c).to(device), (0, pw, 0, ph))[None]
    g = torch.Generator(device=device); g.manual_seed(seed)
    z = torch.randn((1, P) + C_.shape[2:], generator=g, device=device) * noise
    with amp(device):
        for k in range(steps):
            t = torch.full((1,), k / steps, device=device)
            z = z + net(torch.cat([z, C_], 1), t).float() / steps
    r = (z[0, :, :h, :w] * torch.from_numpy(rsd).to(device)[:, None, None]).cpu().numpy().astype(np.float64)
    if inside is not None: r = r * inside[None]
    return (mu + remove_cell_mean(r, lab)).astype(np.float32)
