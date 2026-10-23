"""Prepare a benchmark case from a Xenium output bundle with multimodal cell-segmentation stains.

Grid = pyramid level 2 of morphology_focus (0.85 um/px), same as the kidney training data.
Truth = per-cell mean of the stain channels: 0001 boundary (ATP1A1 / E-cadherin / CD45 in one channel),
0003 interior protein (alphaSMA / Vimentin in one channel), 0002 18S (RNA, kept for reference).
Transcripts (qv >= 20, real genes only) are kept only for genes in the union of the kidney model vocabularies;
the full cell x gene count matrix (all panel genes) comes from cell_feature_matrix.h5 for the ridge / own-mRNA baselines.

usage: prep_case_xenium.py OUTS_DIR CASE_DIR NAME TISSUE PANEL
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd, tifffile, h5py, cv2, scipy.sparse as sp
from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes

outs, cd, name, tissue_name, panel = sys.argv[1:6]
os.makedirs(cd, exist_ok=True)
LEVEL, PX = 2, 0.2125
UM = PX * 2 ** LEVEL
R, C = 'data/xenium_protein_rcc/pilot_L2', 'data/xenium_protein_ccrcc/pilot_L2'
VOCAB = sorted(set(json.load(open(f'{R}/meta.json'))['genes']) | set(json.load(open(f'{C}/meta.json'))['genes']))


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def read_level(path, level):
    with tifffile.TiffFile(path, is_ome=False) as tf:
        s = tf.series[0]
        lv = s.levels[level] if level < len(s.levels) else None
        if lv is None: raise ValueError(f'{path}: only {len(s.levels)} levels')
        return lv.asarray()


def focus(idx):
    fs = sorted(glob.glob(f'{outs}/morphology_focus/*{idx:04d}*.ome.tif'))
    if not fs: raise FileNotFoundError(f'no morphology_focus channel {idx} in {outs}')
    return fs[0]


log(name, 'reading morphology_focus level', LEVEL)
dapi = read_level(focus(0), LEVEL).astype(np.float32)
H, W = dapi.shape
stains = {}
for idx, key in [(1, 'boundary'), (2, 'rna18s'), (3, 'interior')]:
    a = read_level(focus(idx), LEVEL).astype(np.float32)
    assert a.shape == (H, W), (key, a.shape, (H, W))
    stains[key] = a
    log(key, os.path.basename(focus(idx)), 'p50/p99', float(np.percentile(a, 50)), float(np.percentile(a, 99)))

# tissue mask + DAPI normalisation (log1p, z-scored over tissue), as for the kidney inputs
d = gaussian_filter(dapi, 4)
tissue = binary_fill_holes(binary_closing(d > np.percentile(d[d > 0], 20), iterations=5))
ld = np.log1p(dapi)
dz = (ld - ld[tissue].mean()) / (ld[tissue].std() + 1e-4)
np.save(f'{cd}/dapi.npy', dz.astype(np.float16)); np.save(f'{cd}/tissue.npy', tissue)

# transcripts
cols = ['x_location', 'y_location', 'feature_name', 'qv']
tx = pd.read_parquet(f'{outs}/transcripts.parquet', columns=cols)
fn = tx['feature_name']
fn = fn.str.decode('utf-8') if fn.dtype == object and len(fn) and isinstance(fn.iloc[0], bytes) else fn.astype(str)
bad = fn.str.startswith(('NegControl', 'BLANK', 'Unassigned', 'Deprecated', 'Intergenic', 'antisense'))
ok = (tx['qv'].values >= 20) & ~bad.values
n_all = int(ok.sum())
panel_genes = sorted(set(fn[ok].unique()))
genes = [g for g in VOCAB if g in set(panel_genes)]
gpos = {g: i for i, g in enumerate(genes)}
gidx = fn.map(gpos).values
keep = ok & ~pd.isna(gidx)
gx = np.clip((tx['x_location'].values[keep] / UM).astype(np.int32), 0, W - 1)
gy = np.clip((tx['y_location'].values[keep] / UM).astype(np.int32), 0, H - 1)
np.savez(f'{cd}/tx.npz', gy=gy, gx=gx, gidx=gidx[keep].astype(np.int32))
json.dump(genes, open(f'{cd}/genes.json', 'w'))
log('transcripts qv>=20', n_all, 'in model vocab', int(keep.sum()), f'({keep.sum() / max(n_all, 1):.1%})',
    'panel genes', len(panel_genes), 'vocab genes present', len(genes), 'of', len(VOCAB))
del tx, fn

# cell labels from cell boundaries (sub-pixel polygon fill)
cb = pd.read_parquet(f'{outs}/cell_boundaries.parquet')
ids = cb['cell_id'].values
ids = np.array([i.decode() if isinstance(i, bytes) else str(i) for i in ids]) if ids.dtype == object else ids.astype(str)
starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); ends = np.r_[starts[1:], len(ids)]
SH = 4
vx = np.round(cb['vertex_x'].values / UM * (1 << SH)).astype(np.int32)
vy = np.round(cb['vertex_y'].values / UM * (1 << SH)).astype(np.int32)
lab = np.zeros((H, W), np.int32)
cell_ids = ids[starts]
for k, (a, b) in enumerate(zip(starts, ends), start=1):
    cv2.fillPoly(lab, [np.stack([vx[a:b], vy[a:b]], 1)], int(k), lineType=cv2.LINE_8, shift=SH)
np.save(f'{cd}/lab.npy', lab)
n = len(cell_ids) + 1
area = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
log('cells', len(cell_ids), 'with pixels', int((area[1:] > 0).sum()), 'median area px', float(np.median(area[1:][area[1:] > 0])))

cen = pd.DataFrame(dict(c=np.repeat(np.arange(len(starts)), ends - starts), x=cb['vertex_x'].values, y=cb['vertex_y'].values)).groupby('c').mean()
cells = pd.DataFrame(dict(label=np.arange(1, n), cell_id=cell_ids, area_px=area[1:], x_um=cen['x'].values, y_um=cen['y'].values))
for key, a in list(stains.items()) + [('dapi', dapi)]:
    cells[f'truth_{key}'] = np.bincount(lab.ravel(), weights=a.ravel().astype(np.float64), minlength=n)[1:] / np.maximum(area[1:], 1)

# full count matrix from cell_feature_matrix.h5 (gene features only), rows aligned to cells
with h5py.File(f'{outs}/cell_feature_matrix.h5', 'r') as f:
    m = f['matrix']
    bc = np.array([b.decode() for b in m['barcodes'][:]])
    ftype = np.array([t.decode() for t in m['features']['feature_type'][:]])
    fname = np.array([t.decode() for t in m['features']['name'][:]])
    X = sp.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=tuple(m['shape'][:])).T.tocsr()  # cells x features
gsel = ftype == 'Gene Expression'
X = X[:, gsel]; fname = fname[gsel]
row = pd.Series(np.arange(len(bc)), index=bc).reindex(cell_ids)
have = row.notna().values
Xe = sp.vstack([X, sp.csr_matrix((1, X.shape[1]), dtype=X.dtype)]).tocsr()  # last row = empty, for cells missing in h5
Xc = Xe[np.where(have, row.fillna(-1).values, X.shape[0]).astype(np.int64)]
sp.save_npz(f'{cd}/counts.npz', Xc.astype(np.float32))
json.dump(list(fname), open(f'{cd}/counts_genes.json', 'w'))
cells['total_counts'] = np.asarray(Xc.sum(1)).ravel()
cells.to_parquet(f'{cd}/cells.parquet')
log('count matrix', Xc.shape, 'cells found in h5', f'{have.mean():.1%}', 'median counts/cell', float(np.median(cells['total_counts'])))

meta = dict(dataset=name, platform='xenium', tissue=tissue_name, panel=panel, um_per_px=UM, H=H, W=W,
            n_cells=len(cell_ids), n_tx_all=n_all, n_tx_vocab=int(keep.sum()), panel_genes=len(panel_genes),
            vocab_genes_present=len(genes),
            markers={'interior_aSMA_Vim': dict(truth='truth_interior', components=['alphaSMA', 'Vimentin'], own=['ACTA2', 'VIM']),
                     'boundary_ATP1A1_Ecad_CD45': dict(truth='truth_boundary', components=['E-Cadherin', 'CD45'],
                                                       own=['ATP1A1', 'CDH1', 'PTPRC'], note='ATP1A1 is not predicted by the model')})
json.dump(meta, open(f'{cd}/meta.json', 'w'), indent=1)
log('saved', cd)
