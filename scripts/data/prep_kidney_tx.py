"""Transcript table on the kidney training grid, block-sorted for fast rectangular lookup.

Output {pilot_dir}/tx.npz: gy, gx (int16 pixel coords), gid (int16 gene index into meta['genes']),
nuc (uint8, transcript overlaps its nucleus, Xenium overlaps_nucleus),
offsets (int64, len nby*nbx+1: transcripts of block k are [offsets[k], offsets[k+1])), B, nbx, nby.
usage: prep_kidney_tx.py data/<sample>/pilot_L2
"""
import json, sys
import numpy as np, pandas as pd

sd = sys.argv[1]
root = sd.rsplit('/', 1)[0] + '/outs'
meta = json.load(open(f'{sd}/meta.json'))
H, W, ds = meta['H'], meta['W'], 2 ** meta['level']
gset = {g: i for i, g in enumerate(meta['genes'])}

tx = pd.read_parquet(f'{root}/transcripts.parquet', columns=['x_location', 'y_location', 'feature_name', 'qv', 'overlaps_nucleus'])
fn = tx['feature_name'].astype(str)
keep = (tx['qv'].values >= 20) & fn.isin(gset).values
gid = fn[keep].map(gset).values.astype(np.int16)
nuc = tx['overlaps_nucleus'].values[keep].astype(np.uint8)
px = np.clip((tx['x_location'].values[keep] / (0.2125 * ds)).astype(np.int32), 0, W - 1)
py = np.clip((tx['y_location'].values[keep] / (0.2125 * ds)).astype(np.int32), 0, H - 1)

B = 64
nbx, nby = (W + B - 1) // B, (H + B - 1) // B
blk = (py // B) * nbx + (px // B)
o = np.argsort(blk, kind='stable')
offsets = np.searchsorted(blk[o], np.arange(nby * nbx + 1)).astype(np.int64)
np.savez(f'{sd}/tx.npz', gy=py[o].astype(np.int16), gx=px[o].astype(np.int16), gid=gid[o], nuc=nuc[o],
         offsets=offsets, B=B, nbx=nbx, nby=nby)
print('transcripts', len(gid), 'genes', len(gset), 'blocks', nby * nbx, 'nuclear frac', round(nuc.mean(), 3))
