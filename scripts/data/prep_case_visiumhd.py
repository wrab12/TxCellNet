"""Prepare a benchmark case from a Visium HD dataset whose full-resolution microscope image is same-section IF.

Molecules: 2 um bins (square_002um), every UMI of a model-vocabulary gene becomes one pseudo-transcript placed uniformly
at random inside its bin. Image: the IF tissue image (DAPI + antibody channels), resampled to the 0.85 um/px grid.
Cells: Cellpose 'nuclei' on the grid DAPI, expanded by 3 um (no vendor segmentation for these runs).
Truth: per-cell mean of each antibody channel. Counts: bins summed per cell (all genes) for the baselines.

usage: prep_case_visiumhd.py DATA_DIR CASE_DIR NAME TISSUE 'CHANNEL_MAP_JSON'
  CHANNEL_MAP_JSON maps image channel index -> name, e.g. '{"0": "DAPI", "1": "CD45RO", "2": "CD20"}'
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd, tifffile, h5py, cv2, scipy.sparse as sp
from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes
from skimage.segmentation import expand_labels

dd, cd, name, tissue_name, chmap = sys.argv[1:6]
chmap = {int(k): v for k, v in json.loads(chmap).items()}
os.makedirs(cd, exist_ok=True)
UM = 0.85
R, C = 'data/xenium_protein_rcc/pilot_L2', 'data/xenium_protein_ccrcc/pilot_L2'
VOCAB = sorted(set(json.load(open(f'{R}/meta.json'))['genes']) | set(json.load(open(f'{C}/meta.json'))['genes']))
PROT_OF = {'CD20': 'CD20', 'CD45RO': 'CD45RO', 'CD45': 'CD45', 'aSMA': 'alphaSMA', 'alphaSMA': 'alphaSMA', 'SMA': 'alphaSMA',
           'CD3': 'CD3E', 'CD68': 'CD68', 'Vimentin': 'Vimentin', 'PanCK': 'PanCK'}
OWN = {'CD20': ['MS4A1'], 'CD45RO': ['PTPRC'], 'CD45': ['PTPRC'], 'alphaSMA': ['ACTA2'], 'CD3E': ['CD3E'], 'CD68': ['CD68'],
       'Vimentin': ['VIM'], 'PanCK': ['KRT8', 'KRT18', 'KRT19']}


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


b2 = glob.glob(f'{dd}/**/square_002um', recursive=True)[0]
sf = json.load(open(f'{b2}/spatial/scalefactors_json.json'))
mpp = float(sf['microns_per_pixel'])
log(name, 'square_002um at', b2, 'microns_per_pixel', mpp)

# ---- IF image -> grid ----
img_path = [p for p in glob.glob(f'{dd}/tissue_image.*') if p.endswith(('.tif', '.btf', '.tiff'))][0]
with tifffile.TiffFile(img_path) as tf:
    s = tf.series[0]; log('image', s.shape, s.axes, s.dtype, 'levels', len(s.levels))
    arr = s.levels[0].asarray()
if arr.ndim == 3 and arr.shape[-1] <= 8 and arr.shape[0] > 8: arr = np.moveaxis(arr, -1, 0)  # YXC -> CYX
Hf, Wf = arr.shape[1:]
H, W = int(round(Hf * mpp / UM)), int(round(Wf * mpp / UM))
chans = {chmap[i]: cv2.resize(arr[i].astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) for i in chmap}
del arr
for k, v in chans.items(): log('channel', k, 'p50/p99', float(np.percentile(v, 50)), float(np.percentile(v, 99)))
dapi = chans['DAPI']
d = gaussian_filter(dapi, 4)
tissue = binary_fill_holes(binary_closing(d > np.percentile(d[d > 0], 20), iterations=5))
ld = np.log1p(dapi)
np.save(f'{cd}/dapi.npy', ((ld - ld[tissue].mean()) / (ld[tissue].std() + 1e-4)).astype(np.float16)); np.save(f'{cd}/tissue.npy', tissue)

# ---- cells: Cellpose nuclei on the grid DAPI, expanded by 3 um ----
from cellpose import models
cp = models.Cellpose(gpu=True, model_type='nuclei')
x = np.clip(dapi / np.percentile(dapi[tissue], 99.5), 0, 1).astype(np.float32)
masks = np.zeros((H, W), np.int32); T, O, nxt = 4096, 64, 0
for y0 in range(0, H, T):  # tile to bound memory; labels made unique across tiles
    for x0 in range(0, W, T):
        ya, xa, yb, xb = max(y0 - O, 0), max(x0 - O, 0), min(y0 + T + O, H), min(x0 + T + O, W)
        m = cp.eval(x[ya:yb, xa:xb], diameter=8.0 / UM, channels=[0, 0])[0].astype(np.int32)
        core = m[y0 - ya:y0 - ya + min(T, H - y0), x0 - xa:x0 - xa + min(T, W - x0)]
        core = np.where(core > 0, core + nxt, 0); nxt = int(core.max()) if core.max() > 0 else nxt
        masks[y0:y0 + core.shape[0], x0:x0 + core.shape[1]] = core
masks[~tissue] = 0
lab = expand_labels(masks, distance=int(round(3.0 / UM))).astype(np.int32)
u, lab = np.unique(lab, return_inverse=True); lab = lab.reshape(H, W).astype(np.int32)
if u[0] != 0: lab += 1
np.save(f'{cd}/lab.npy', lab)
n = int(lab.max()) + 1
area = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
log('cells', n - 1, 'median area px', float(np.median(area[1:])))

# ---- molecules from 2 um bins ----
with h5py.File(f'{b2}/filtered_feature_bc_matrix.h5', 'r') as f:
    m = f['matrix']
    bcs = np.array([b.decode() for b in m['barcodes'][:]])
    gnames = np.array([t.decode() for t in m['features']['name'][:]])
    M = sp.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=tuple(m['shape'][:]))  # genes x bins
pos = pd.read_parquet(f'{b2}/spatial/tissue_positions.parquet').set_index('barcode').loc[bcs]
by = pos['pxl_row_in_fullres'].values * mpp / UM; bx = pos['pxl_col_in_fullres'].values * mpp / UM
half = 1.0 / UM  # 2 um bin -> +-1 um jitter
genes = [g for g in VOCAB if g in set(gnames)]
gpos = {g: i for i, g in enumerate(genes)}
rows_v = np.array([gpos.get(g, -1) for g in gnames])
Mv = M[rows_v >= 0].tocoo()  # vocab genes x bins
gi = rows_v[rows_v >= 0][Mv.row]; cnt = Mv.data.astype(np.int64)
rng = np.random.default_rng(0)
rep_b = np.repeat(Mv.col, cnt); rep_g = np.repeat(gi, cnt)
ty = np.clip((by[rep_b] + rng.uniform(-half, half, len(rep_b))).astype(np.int32), 0, H - 1)
tx = np.clip((bx[rep_b] + rng.uniform(-half, half, len(rep_b))).astype(np.int32), 0, W - 1)
np.savez(f'{cd}/tx.npz', gy=ty, gx=tx, gidx=rep_g.astype(np.int32))
json.dump(genes, open(f'{cd}/genes.json', 'w'))
log('bins', len(bcs), 'UMIs total', int(M.sum()), 'in vocab', len(rep_b), 'vocab genes present', len(genes))

# ---- per-cell counts (all genes) by bin centre ----
bl = lab[np.clip(by.astype(int), 0, H - 1), np.clip(bx.astype(int), 0, W - 1)]
A = sp.csr_matrix((np.ones(len(bl)), (bl, np.arange(len(bl)))), shape=(n, len(bl)))
Xc = (A @ M.T.tocsc()).tocsr()[1:].astype(np.float32)
sp.save_npz(f'{cd}/counts.npz', Xc); json.dump(list(gnames), open(f'{cd}/counts_genes.json', 'w'))

yy = np.bincount(lab.ravel(), weights=np.repeat(np.arange(H, dtype=np.float64), W), minlength=n) / np.maximum(area, 1)
xx = np.bincount(lab.ravel(), weights=np.tile(np.arange(W, dtype=np.float64), H), minlength=n) / np.maximum(area, 1)
cells = pd.DataFrame(dict(label=np.arange(1, n), cell_id=[f'c{i}' for i in range(1, n)], area_px=area[1:], x_um=xx[1:] * UM, y_um=yy[1:] * UM,
                          total_counts=np.asarray(Xc.sum(1)).ravel()))
markers = {}
for k, v in chans.items():
    cells[f'truth_{k}'] = np.bincount(lab.ravel(), weights=v.ravel().astype(np.float64), minlength=n)[1:] / np.maximum(area[1:], 1)
    if k in PROT_OF:
        p = PROT_OF[k]
        markers[k] = dict(truth=f'truth_{k}', components=[p], own=OWN.get(p, []))
cells.to_parquet(f'{cd}/cells.parquet')
meta = dict(dataset=name, platform='visium_hd', tissue=tissue_name, panel='whole_transcriptome_probe', um_per_px=UM, H=H, W=W,
            n_cells=n - 1, n_tx_vocab=int(len(rep_b)), vocab_genes_present=len(genes), markers=markers,
            notes='pseudo-transcripts from 2 um bins; cells = Cellpose nuclei + 3 um')
json.dump(meta, open(f'{cd}/meta.json', 'w'), indent=1)
log('saved', cd, 'median counts/cell', float(np.median(cells['total_counts'])))
