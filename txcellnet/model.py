"""TxCellNet stage 1: conditional-mean protein image from transcripts, DAPI and cell segmentation.

Every molecule is replaced by a learnable embedding of its gene and the embeddings are summed at the molecule's pixel
(gene field), together with log(1 + molecule count). DAPI and the gene field enter a U-Net (widths 64, 128, 256, 512).
A cell-neighbourhood transformer pools the U-Net features and the gene field into one token per cell, exchanges
information between the cells of the window with distance-biased attention (2 layers, 4 heads, length scales initialised
to 6, 12, 24 and 48 px) and adds each cell's update back to its pixels; a 1x1 convolution gives the P output channels.

cfg (saved next to the weights as cfg.json):
  panel_genes      optional list: panel matching, the model reads only these genes
  nodapi / noniche ablations: DAPI input zeroed / cell-neighbourhood transformer removed
Model interface: prepare(genes) -> list of genes the model reads (their order defines gene ids);
forward(x_dapi, (flat_pixel_index, gene_id, n, h, w), cell_labels) -> (n, P, h, w).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gn_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class UNet(nn.Module):
    def __init__(s, cin, w):
        super().__init__()
        n = len(w) - 1
        s.inp = gn_block(cin, w[0])
        s.down = nn.ModuleList([gn_block(w[i], w[i + 1]) for i in range(n)])
        s.up = nn.ModuleList([nn.ConvTranspose2d(w[i + 1], w[i], 2, stride=2) for i in reversed(range(n))])
        s.dec = nn.ModuleList([gn_block(2 * w[i], w[i]) for i in reversed(range(n))])

    def forward(s, x):
        hs = [s.inp(x)]
        for d in s.down: hs.append(d(F.max_pool2d(hs[-1], 2)))
        h = hs.pop()
        for u, d in zip(s.up, s.dec): h = d(torch.cat([u(h), hs.pop()], 1))
        return h


class CellNiche(nn.Module):
    """Cell-neighbourhood transformer: per-cell tokens (mean features + log pixel count), attention with a Gaussian
    distance bias per head (learned length scale), update broadcast back to the cell's pixels. The output layer is
    zero-initialised, so at the start of training the network is a plain U-Net."""

    def __init__(s, cin, cout, d=128, heads=4, layers=2):
        super().__init__()
        s.heads = heads
        s.inp = nn.Linear(cin + 1, d)
        s.layers = nn.ModuleList([nn.ModuleDict(dict(
            ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), proj=nn.Linear(d, d),
            ln2=nn.LayerNorm(d), mlp=nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))))
            for _ in range(layers)])
        s.log_sigma = nn.Parameter(torch.log(torch.tensor([6., 12., 24., 48.][:heads])))
        s.out = nn.Linear(d, cout)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)

    def forward(s, feat, lab):
        n, c, h, w = feat.shape
        m = (lab > 0).reshape(-1)
        if not m.any(): return torch.zeros(n, s.out.out_features, h, w, device=feat.device)
        key = (lab.long() + torch.arange(n, device=feat.device).view(n, 1, 1) * (1 << 22)).reshape(-1)[m]
        uniq, inv = torch.unique(key, return_inverse=True)
        N = len(uniq)
        cnt = torch.zeros(N, device=feat.device).index_add(0, inv, torch.ones(len(inv), device=feat.device))
        tok = torch.zeros(N, c, device=feat.device).index_add(0, inv, feat.permute(0, 2, 3, 1).reshape(-1, c)[m]) / cnt[:, None]
        yy, xx = torch.meshgrid(torch.arange(h, device=feat.device), torch.arange(w, device=feat.device), indexing='ij')
        pos = torch.stack([yy, xx], -1).reshape(1, h * w, 2).expand(n, -1, -1).reshape(-1, 2)[m].float()
        cen = torch.zeros(N, 2, device=feat.device).index_add(0, inv, pos) / cnt[:, None]
        sid = uniq >> 22
        x = s.inp(torch.cat([tok, torch.log(cnt)[:, None]], 1))
        bias = -torch.cdist(cen, cen).square()[None] / (2 * torch.exp(s.log_sigma)[:, None, None] ** 2)
        bias = bias.masked_fill((sid[:, None] != sid[None, :])[None], float('-inf'))[None]
        for L in s.layers:
            q, k, v = L['qkv'](L['ln1'](x)).view(N, 3, s.heads, -1).permute(1, 2, 0, 3)[:, None]
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)[0]
            x = x + L['proj'](a.permute(1, 0, 2).reshape(N, -1))
            x = x + L['mlp'](L['ln2'](x))
        upd = s.out(x)
        add = torch.zeros(n * h * w, upd.shape[1], device=feat.device).index_put((m.nonzero()[:, 0],), upd[inv])
        return add.view(n, h, w, -1).permute(0, 3, 1, 2)


def gauss_kernel(sigma):
    r = int(3 * sigma + 0.5); t = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); return k / k.sum()


class GenNet(nn.Module):
    """prots: output channel names; own_genes: protein -> coding genes (kept for reference, not read by the network);
    cfg: configuration dict (see the module docstring); train_genes: gene vocabulary (one learned embedding each)."""

    def __init__(s, prots, own_genes, cfg, train_genes, GD=64):
        super().__init__()
        if cfg.get('rep', 'learned') != 'learned' or cfg.get('enc', 'sum') != 'sum' or cfg.get('own', 0):
            raise ValueError('only the configuration of the paper is supported: rep=learned, enc=sum, own=0')
        s.cfg, s.prots, s.own_genes, s.GD = cfg, prots, own_genes, GD
        s.train_genes = list(train_genes)
        s.emb = nn.Embedding(len(train_genes), GD); nn.init.normal_(s.emb.weight, std=0.1)
        cin = 1 + GD + 1
        w = (64, 128, 256, 512)
        s.unet = UNet(cin, w)
        s.cell = CellNiche(w[0] + GD + 1, w[0])
        s.head = nn.Conv2d(w[0], len(prots), 1)
        s.prepare(train_genes)

    def prepare(s, genes=None):
        """Gene table of the model. Genes outside the training vocabulary cannot be read. With panel matching, genes outside
        the target panel keep their table slot but are renamed, so that no panel symbol maps to them (never read, in
        training or at inference). The `genes` argument is accepted for interface compatibility and not used."""
        keep = set(s.cfg['panel_genes']) if s.cfg.get('panel_genes') else None
        vocab = [g if keep is None or g in keep else f'__off__{g}' for g in s.train_genes]
        s._vocab = vocab
        return vocab

    def encode(s, flat, gid, n, h, w):
        """gene field: sum of the gene embeddings of the molecules at each pixel, plus log(1 + molecule count)"""
        E = s.emb.weight
        src = torch.cat([E[gid].float(), torch.ones(len(gid), 1, device=E.device)], 1)
        z = torch.zeros(n * h * w, src.shape[1], device=E.device).index_add(0, flat, src)
        z = z.view(n, h, w, -1).permute(0, 3, 1, 2)
        return torch.cat([z[:, :-1], torch.log1p(z[:, -1:])], 1)

    def forward(s, x, tx, lab):
        g = s.encode(*tx)
        if s.cfg.get('nodapi', 0): x = torch.zeros_like(x)
        h = s.unet(torch.cat([x, g], 1))
        if s.cfg.get('noniche', 0): return s.head(h.float())
        with torch.autocast(h.device.type, enabled=False):  # the transformer runs in fp32
            h = h.float() + s.cell(torch.cat([h.float(), g.float()], 1), lab)
        return s.head(h)
