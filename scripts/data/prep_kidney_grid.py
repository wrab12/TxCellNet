"""Training grid of one kidney Xenium Protein section (pyramid level 2 of morphology_focus = 0.85 um/px).

Writes data/<sample>/pilot_L2/:
  X.npy        float16 (1, H, W)   log1p(DAPI)
  Y.npy        float16 (27, H, W)  log1p(max(protein - 100, 0)) of the 27 immunofluorescence channels
  valid.npy    bool (H, W)         no saturation-masked pixel in any channel (raw value >= 90 everywhere)
  tissue.npy   bool (H, W)         DAPI tissue mask
  cells.npy    int32 (H, W)        cell labels rasterised from cell_boundaries.parquet; cell_ids.csv: label - 1 -> cell id
  meta.json    prots, genes (gene_panel.json), own_genes (protein -> coding genes), level, um_per_px, H, W
usage: prep_kidney_grid.py SAMPLE [LEVEL=2]      (SAMPLE: xenium_protein_rcc or xenium_protein_ccrcc; reads data/SAMPLE/outs)
"""
import json, os, sys, time
import h5py, numpy as np, pandas as pd, tifffile
from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes
from skimage.draw import polygon as draw_polygon

from txcellnet.genes import own_cols

PX = 0.2125  # um per full-res pixel


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def read_level(path, level):
    with tifffile.TiffFile(path, is_ome=False) as tf:  # OME metadata links all channels; read this file only
        s = tf.series[0]
        lv = s.levels[level] if level < len(s.levels) else None
        if lv is None: raise ValueError(f'{path}: only {len(s.levels)} levels')
        return lv.asarray()


def main(sample, level=2):
    root = f'data/{sample}/outs'
    out = f'data/{sample}/pilot_L{level}'
    os.makedirs(out, exist_ok=True)
    ds = 2 ** level
    mf = sorted(os.listdir(f'{root}/morphology_focus'))
    with h5py.File(f'{root}/cell_feature_matrix.h5', 'r') as f:
        m = f['matrix']
        fname_all = np.array([x.decode() for x in m['features']['name'][:]])
        ftype_all = np.array([x.decode() for x in m['features']['feature_type'][:]])
    prots = list(fname_all[ftype_all == 'Protein Expression'])

    def fname(p):  # protein short name -> morphology_focus file
        key = p.lower().replace(' ', '')
        hits = [f for f in mf if f.split('_', 1)[1].rsplit('.ome', 1)[0] == key]
        assert len(hits) == 1, (p, hits)
        return hits[0]

    log('reading DAPI')
    dapi = read_level(f'{root}/morphology_focus/ch0000_dapi.ome.tif', level).astype(np.float32)
    H, W = dapi.shape
    log('grid', H, W, 'um/px', PX * ds)

    T = np.zeros((len(prots), H, W), np.float16)
    valid = np.ones((H, W), bool)
    for j, p in enumerate(prots):
        a = read_level(f'{root}/morphology_focus/{fname(p)}', level).astype(np.float32)
        valid &= a >= 90  # saturation-masked pixels are 0 before the +100 offset
        T[j] = np.log1p(np.clip(a - 100, 0, None))
        log('protein', p, fname(p), 'p99', float(np.percentile(T[j], 99)))

    # tissue mask from DAPI
    d = gaussian_filter(dapi, 4)
    tissue = binary_fill_holes(binary_closing(d > np.percentile(d[d > 0], 20), iterations=5))

    genes = list(dict.fromkeys(g['type']['data']['name'] for g in json.load(open(f'{root}/gene_panel.json'))['payload']['targets']
                               if g['type']['descriptor'] == 'gene'))
    own_idx = [own_cols(genes, p) for p in prots]
    Xin = np.log1p(dapi)[None].astype(np.float16)  # the network reads DAPI and the transcripts (prep_kidney_tx.py)

    log('rasterising cells')
    cb = pd.read_parquet(f'{root}/cell_boundaries.parquet')
    lab = np.zeros((H, W), np.int32)
    ids = cb['cell_id'].astype(str).values
    uid, first = np.unique(ids, return_index=True)
    order = np.argsort(first); uid = uid[order]
    starts = np.sort(first); ends = np.r_[starts[1:], len(ids)]
    vx = cb['vertex_x'].values / (PX * ds); vy = cb['vertex_y'].values / (PX * ds)
    for n, (a, b) in enumerate(zip(starts, ends), start=1):
        rr, cc = draw_polygon(vy[a:b], vx[a:b], shape=(H, W))
        lab[rr, cc] = n
    pd.Series(uid, name='cell_id').to_csv(f'{out}/cell_ids.csv', index_label='label_minus_1')

    np.save(f'{out}/X.npy', Xin); np.save(f'{out}/Y.npy', T)
    np.save(f'{out}/valid.npy', valid); np.save(f'{out}/tissue.npy', tissue); np.save(f'{out}/cells.npy', lab)
    json.dump(dict(prots=prots, genes=genes, level=level, um_per_px=PX * ds, H=H, W=W,
                   own_genes={p: [genes[i] for i in idx] for p, idx in zip(prots, own_idx)}),
              open(f'{out}/meta.json', 'w'), indent=1)
    log('saved', out)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
