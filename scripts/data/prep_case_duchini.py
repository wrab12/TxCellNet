"""Prepare benchmark cases from Duchini et al. 2026 (Xenium Prime 5K, then COMET on the same section; 4 TMA cores).

Grid and inputs: 0.85 um/px grid over the transcript extent (Xenium um space), cell labels from
Xenium cell boundaries, DAPI = COMET DAPI warped onto the grid with the provided affine (inverse of COMET px -> Xenium px).
Truth = the authors' per-cell COMET mean intensities (Cell_quantifications, keyed by Xenium cell id);
counts = the per-cell gene counts of the same table.

usage: prep_case_duchini.py CORE CASE_DIR NAME     (CORE e.g. core03_tonsil)
"""
import json, os, sys, time, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, tifffile, cv2, scipy.sparse as sp
from scipy.ndimage import gaussian_filter, map_coordinates

core, cd, name = sys.argv[1:4]
os.makedirs(cd, exist_ok=True)
DU = 'data/duchini_xenium_comet/ex'
UM, XPX = 0.85, 0.2125
R, C = 'data/xenium_protein_rcc/pilot_L2', 'data/xenium_protein_ccrcc/pilot_L2'
VOCAB = sorted(set(json.load(open(f'{R}/meta.json'))['genes']) | set(json.load(open(f'{C}/meta.json'))['genes']))
MARK = {'CD20': 'CD20 - TRITC', 'CD45': 'CD45 - TRITC', 'CD45RA': 'CD45RA - TRITC', 'E-Cadherin': 'E-Cadherin - TRITC',
        'Vimentin': 'vimentin - TRITC', 'alphaSMA': 'αSMA - TRITC', 'PanCK': 'CK8 - TRITC'}
OWN = {'CD20': ['MS4A1'], 'CD45': ['PTPRC'], 'CD45RA': ['PTPRC'], 'E-Cadherin': ['CDH1'], 'Vimentin': ['VIM'],
       'alphaSMA': ['ACTA2'], 'PanCK': ['KRT8']}
TISSUE = {'core01_hcc': 'liver_HCC', 'core02_non_tumour': 'liver_normal', 'core03_tonsil': 'tonsil', 'core04_hca': 'liver_HCA'}


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


xd = f'{DU}/UnumLocalia/{core}/xenium'
tx = pd.read_parquet(f'{xd}/transcripts.parquet', columns=['feature_name', 'x_location', 'y_location', 'qv'])
tx = tx[tx['qv'] >= 20]
x0, y0 = tx['x_location'].min() - 20, tx['y_location'].min() - 20
W = int(np.ceil((tx['x_location'].max() + 20 - x0) / UM)); H = int(np.ceil((tx['y_location'].max() + 20 - y0) / UM))
fn = tx['feature_name'].astype(str)
genes = [g for g in VOCAB if g in set(fn.unique())]; gpos = {g: i for i, g in enumerate(genes)}
gi = fn.map(gpos).values; k = ~pd.isna(gi)
gx = np.clip(((tx['x_location'].values - x0) / UM).astype(np.int32), 0, W - 1); gy = np.clip(((tx['y_location'].values - y0) / UM).astype(np.int32), 0, H - 1)
np.savez(f'{cd}/tx.npz', gy=gy[k], gx=gx[k], gidx=gi[k].astype(np.int32)); json.dump(genes, open(f'{cd}/genes.json', 'w'))
log(name, 'grid', H, W, 'transcripts', len(tx), 'in vocab', int(k.sum()), 'vocab genes', len(genes))

cb = pd.read_parquet(f'{xd}/cell_boundaries.parquet')
ids = cb['cell_id'].astype(str).values
starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); ends = np.r_[starts[1:], len(ids)]
SH = 4
vx = np.round((cb['vertex_x'].values - x0) / UM * (1 << SH)).astype(np.int32); vy = np.round((cb['vertex_y'].values - y0) / UM * (1 << SH)).astype(np.int32)
lab = np.zeros((H, W), np.int32)
for j, (a, b) in enumerate(zip(starts, ends), start=1):
    cv2.fillPoly(lab, [np.stack([vx[a:b], vy[a:b]], 1)], j, lineType=cv2.LINE_8, shift=SH)
np.save(f'{cd}/lab.npy', lab)
cell_ids = ids[starts]; n = len(cell_ids) + 1
area = np.bincount(lab.ravel(), minlength=n).astype(np.float64)

M = np.loadtxt(f'{DU}/UnumLocalia/{core}/comet/matrix_comet.csv', delimiter=','); Mi = np.linalg.inv(M)
with tifffile.TiffFile(f'{DU}/COMET_ome_tiff/{core[:6]}_comet.ome.tif') as tf:
    names = [e.attrib.get('Name', '') for e in ET.fromstring(tf.ome_metadata).iter() if e.tag.endswith('Channel')]
    arr = tf.series[0].levels[0].asarray()
di = [i for i, nm in enumerate(names) if 'DAPI' in nm.upper()]
dapi_c = arr[di[0] if di else 0].astype(np.float32); del arr
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
px_x, px_y = (x0 + (xx + 0.5) * UM) / XPX, (y0 + (yy + 0.5) * UM) / XPX
cxp = Mi[0, 0] * px_x + Mi[0, 1] * px_y + Mi[0, 2]; cyp = Mi[1, 0] * px_x + Mi[1, 1] * px_y + Mi[1, 2]
dapi = map_coordinates(gaussian_filter(dapi_c, 1.2), [cyp, cxp], order=1, mode='constant', cval=0)
d = np.log1p(dapi); tis = d > np.percentile(d[d > 0], 20)
np.save(f'{cd}/dapi.npy', ((d - d[tis].mean()) / (d[tis].std() + 1e-4)).astype(np.float16))

q = pd.read_csv(f'{DU}/Cell_quantifications/{core}_quantification_cells.csv')
q = q.set_index(q['cell_id'].astype(str))
qi = q.reindex(cell_ids)
has = qi['label'].notna().values if 'label' in qi.columns else qi.iloc[:, 0].notna().values
gene_cols = list(q.columns[q.columns.get_loc('CD206 - TRITC_mean') + 1:])
Xc = sp.csr_matrix(np.nan_to_num(qi[gene_cols].values.astype(np.float32)))
sp.save_npz(f'{cd}/counts.npz', Xc); json.dump(gene_cols, open(f'{cd}/counts_genes.json', 'w'))
cen = pd.DataFrame(dict(c=np.repeat(np.arange(len(starts)), ends - starts), x=cb['vertex_x'].values, y=cb['vertex_y'].values)).groupby('c').mean()
cells = pd.DataFrame(dict(label=np.arange(1, n), cell_id=cell_ids, area_px=area[1:], x_um=cen['x'].values, y_um=cen['y'].values,
                          has_truth=has, total_counts=np.asarray(Xc.sum(1)).ravel()))
markers = {}
for p, col in MARK.items():
    cells[f'truth_{p}'] = qi[f'{col}_mean'].values.astype(np.float64)
    markers[p] = dict(truth=f'truth_{p}', components=[p], own=OWN[p])
cells.to_parquet(f'{cd}/cells.parquet')
meta = dict(dataset=name, platform='xenium', tissue=TISSUE.get(core, core), panel='prime5k', protein_platform='COMET',
            um_per_px=UM, H=H, W=W, n_cells=n - 1, vocab_genes_present=len(genes), eval_mask='has_truth', markers=markers,
            notes='DAPI from COMET (warped); truth = authors per-cell COMET means')
json.dump(meta, open(f'{cd}/meta.json', 'w'), indent=1)
log('saved', cd, 'cells', n - 1, 'with COMET truth', int(has.sum()))
