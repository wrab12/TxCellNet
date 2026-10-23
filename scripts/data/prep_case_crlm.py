"""Prepare benchmark cases from Eapen et al. 2026 (colorectal liver metastasis): Xenium (462-gene custom panel), then IMC
(38-plex) on the same FFPE section. One case per IMC ROI.

Registration (IMC DNA1+DNA2 -> Xenium DAPI):
  grid = Xenium pyramid level 2 (0.85 um/px); DAPI = max projection of the z-stack (no morphology_focus in the GEO files)
  1) coarse: normalised template matching at level 4 (3.4 um/px) over 8 orientations (4 rotations x mirror)
  2) fine: ECC affine at 0.85 um/px between the DAPI crop and the coarsely warped DNA
Case = ROI bounding box + 150 um margin (context for the model); evaluated cells = centroid inside the ROI (10 um inset).
Truth = per-cell mean of each IMC channel warped onto the grid.

usage: prep_case_crlm.py SAMPLE ROI_NPZ CASE_DIR NAME
  SAMPLE e.g. GSM9815454_tumor1_M242T (prefix of the Xenium files in data/crlm_xenium_imc/xenium/ex)
"""
import json, os, sys, time
import numpy as np, pandas as pd, tifffile, h5py, cv2, scipy.sparse as sp
from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes

sample, roi_npz, cd, name = sys.argv[1:5]
os.makedirs(cd, exist_ok=True)
XD = 'data/crlm_xenium_imc/xenium/ex'
PX = 0.2125; UM = PX * 4
R, C = 'data/xenium_protein_rcc/pilot_L2', 'data/xenium_protein_ccrcc/pilot_L2'
VOCAB = sorted(set(json.load(open(f'{R}/meta.json'))['genes']) | set(json.load(open(f'{C}/meta.json'))['genes']))
OWN = json.load(open(f'{R}/meta.json'))['own_genes']


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def zmax(level):
    with tifffile.TiffFile(f'{XD}/{sample}_morphology.ome.tif') as tf:
        return tf.series[0].levels[level].asarray().max(0).astype(np.float32)


def norm(a, tissue=None):
    a = np.log1p(np.clip(a, 0, None)); v = a[tissue] if tissue is not None else a[a > 0]
    return ((a - np.median(v)) / (v.std() + 1e-6)).astype(np.float32)


# ---- load ----
Z = np.load(roi_npz, allow_pickle=False)
chans = [str(c) for c in Z['channels']]; img = Z['img'].astype(np.float32); meta_roi = json.loads(str(Z['meta']))
dna = img[chans.index('DNA1')] + img[chans.index('DNA2')]
log(name, 'ROI', meta_roi['desc'], img.shape, 'channels', len(chans))
D2 = zmax(2); H, W = D2.shape
D4 = zmax(4)
log('Xenium DAPI z-max level2', D2.shape, 'level4', D4.shape)

# ---- coarse: template matching at level 4 over 8 orientations ----
s4 = PX * 16  # um/px at level 4
dna4 = cv2.resize(dna, (int(round(dna.shape[1] / s4)), int(round(dna.shape[0] / s4))), interpolation=cv2.INTER_AREA)
Dn4 = cv2.GaussianBlur(norm(D4), (0, 0), 1.0)
best = None
for flip in (False, True):
    for k in range(4):
        t = np.fliplr(dna4) if flip else dna4
        t = np.ascontiguousarray(np.rot90(t, k))
        tn = cv2.GaussianBlur(norm(t), (0, 0), 1.0)
        if tn.shape[0] >= Dn4.shape[0] or tn.shape[1] >= Dn4.shape[1]: continue
        res = cv2.matchTemplate(Dn4, tn, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if best is None or mx > best[0]: best = (mx, flip, k, loc, tn.shape)
score, flip, k, loc, tshape = best
log('coarse match NCC', round(score, 3), 'flip', flip, 'rot90 x', k, 'loc(level4 px)', loc)


def orient_matrix(h, w, flip, k):
    """3x3 matrix mapping IMC pixel (x, y) to the oriented template pixel (as produced by fliplr then rot90 k times)."""
    M = np.eye(3)
    if flip: M = np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1.0]]) @ M
    hh, ww = h, w
    for _ in range(k):  # np.rot90 (counter-clockwise): new (x', y') = (y, ww-1-x); new shape (ww, hh)
        M = np.array([[0, 1, 0], [-1, 0, ww - 1], [0, 0, 1.0]]) @ M
        hh, ww = ww, hh
    return M


h1, w1 = dna.shape
S_imc_to_4 = np.diag([1 / s4, 1 / s4, 1.0])  # IMC px (1 um) -> level-4 px scale
O = orient_matrix(dna4.shape[0], dna4.shape[1], flip, k)
T4 = np.array([[1, 0, loc[0]], [0, 1, loc[1]], [0, 0, 1.0]])
M4 = T4 @ O @ S_imc_to_4  # IMC px -> level-4 px
M0 = np.diag([4.0, 4.0, 1.0]) @ M4  # -> level-2 px (4x finer)
# small correction: orient_matrix works on the downsampled template grid; the IMC->level4 scaling is applied first, fine for coarse init

# ---- fine: ECC affine at level 2 inside the ROI bounding box ----
corners = np.array([[0, 0, 1], [w1, 0, 1], [0, h1, 1], [w1, h1, 1.0]]).T
cg = (M0 @ corners)[:2]
mg = int(150 / UM)
x0, x1 = int(max(cg[0].min() - mg, 0)), int(min(cg[0].max() + mg, W)); y0, y1 = int(max(cg[1].min() - mg, 0)), int(min(cg[1].max() + mg, H))
Tcrop = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
Dc = norm(D2[y0:y1, x0:x1])
warp0 = cv2.warpAffine(norm(dna), (Tcrop @ M0)[:2], (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=float(norm(dna).min()))
valid0 = (cv2.warpAffine(np.ones_like(dna), (Tcrop @ M0)[:2], (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)
Wm = np.eye(2, 3, dtype=np.float32)
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
try:
    cc, Wm = cv2.findTransformECC(cv2.GaussianBlur(Dc, (0, 0), 2.0), cv2.GaussianBlur(warp0, (0, 0), 2.0), Wm, cv2.MOTION_AFFINE, crit, valid0, 5)
except cv2.error as e:
    cc = float('nan'); log('ECC failed', e)
W3 = np.vstack([Wm, [0, 0, 1]]).astype(np.float64)  # maps crop coords -> warp0 coords
M = np.linalg.inv(W3) @ Tcrop @ M0  # IMC px -> crop px (level 2)
log('ECC cc', round(float(cc), 3), 'final affine', np.round(M[:2], 4).tolist())

# ---- warp all IMC channels onto the crop grid; ROI footprint ----
hc, wc = y1 - y0, x1 - x0
ones = cv2.warpAffine(np.ones((h1, w1), np.float32), M[:2], (wc, hc), flags=cv2.INTER_NEAREST)
inset = cv2.erode((ones > 0.5).astype(np.uint8), np.ones((2 * int(10 / UM) + 1,) * 2, np.uint8)) > 0
warped = {c: cv2.warpAffine(img[i], M[:2], (wc, hc), flags=cv2.INTER_LINEAR) for i, c in enumerate(chans)}
dn = warped['DNA1'] + warped['DNA2']
r_reg = np.corrcoef(norm(dn)[inset].ravel(), Dc[inset].ravel())[0, 1]
log('registered DNA vs DAPI Pearson inside ROI', round(float(r_reg), 3))

# ---- case: DAPI, transcripts, cells, truth, counts ----
tissue_full = binary_fill_holes(binary_closing(gaussian_filter(D2, 4) > np.percentile(D2[D2 > 0], 20), iterations=5))
ld = np.log1p(D2); mu, sdv = ld[tissue_full].mean(), ld[tissue_full].std() + 1e-4
np.save(f'{cd}/dapi.npy', ((ld[y0:y1, x0:x1] - mu) / sdv).astype(np.float16))
cache = f'data/crlm_xenium_imc/xenium/cache/{sample}_tx_vocab.npz'  # qv>=20, model-vocabulary genes only (parquet read is slow)
if not os.path.exists(cache):
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tx = pd.read_parquet(f'{XD}/{sample}_transcripts.parquet', columns=['x_location', 'y_location', 'feature_name', 'qv'])
    fn = tx['feature_name'].astype(str)
    genes_c = [g for g in VOCAB if g in set(fn.unique())]; gp = {g: i for i, g in enumerate(genes_c)}
    gi = fn.map(gp).values; k = (tx['qv'].values >= 20) & ~pd.isna(gi)
    np.savez(cache, x=tx['x_location'].values[k].astype(np.float32), y=tx['y_location'].values[k].astype(np.float32),
             gidx=gi[k].astype(np.int32), genes=np.array(genes_c))
    del tx, fn
Tc = np.load(cache)
genes = [str(g) for g in Tc['genes']]
gx_all = Tc['x'] / UM - x0; gy_all = Tc['y'] / UM - y0
keep = (gx_all >= 0) & (gx_all < wc) & (gy_all >= 0) & (gy_all < hc)
np.savez(f'{cd}/tx.npz', gy=gy_all[keep].astype(np.int32), gx=gx_all[keep].astype(np.int32), gidx=Tc['gidx'][keep].astype(np.int32))
json.dump(genes, open(f'{cd}/genes.json', 'w'))
log('transcripts in crop (vocab)', int(keep.sum()), 'vocab genes present', len(genes))

cb = pd.read_parquet(f'{XD}/{sample}_cell_boundaries.parquet')
ids = cb['cell_id'].astype(str).values
starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); ends = np.r_[starts[1:], len(ids)]
vx = cb['vertex_x'].values / UM - x0; vy = cb['vertex_y'].values / UM - y0
cxm = np.add.reduceat(vx, starts) / (ends - starts); cym = np.add.reduceat(vy, starts) / (ends - starts)
use = (cxm > -20) & (cxm < wc + 20) & (cym > -20) & (cym < hc + 20)
lab = np.zeros((hc, wc), np.int32); SH = 4; cell_ids, cxs, cys = [], [], []
for a, b, u, cx, cy in zip(starts, ends, use, cxm, cym):
    if not u: continue
    cell_ids.append(ids[a]); cxs.append(cx); cys.append(cy)
    pts = np.stack([np.round(vx[a:b] * (1 << SH)), np.round(vy[a:b] * (1 << SH))], 1).astype(np.int32)
    cv2.fillPoly(lab, [pts], len(cell_ids), lineType=cv2.LINE_8, shift=SH)
np.save(f'{cd}/lab.npy', lab)
n = len(cell_ids) + 1
area = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
cxs, cys = np.array(cxs), np.array(cys)
inroi = inset[np.clip(cys.astype(int), 0, hc - 1), np.clip(cxs.astype(int), 0, wc - 1)]
cells = pd.DataFrame(dict(label=np.arange(1, n), cell_id=cell_ids, area_px=area[1:], x_um=(cxs + x0) * UM, y_um=(cys + y0) * UM, in_roi=inroi))
PROT = {'CD20': 'CD20', 'PanCK': 'PanCK', 'VISTA': 'VISTA', 'CD16': 'CD16', 'CD163': 'CD163', 'Vimentin': 'Vimentin', 'Ki-67': 'Ki-67',
        'CD31': 'CD31', 'CD45': 'CD45', 'CD11c': 'CD11c', 'CD4': 'CD4', 'E-Cadherin': 'E-Cadherin', 'CD68': 'CD68', 'CD8A': 'CD8A',
        'PD-1': 'PD-1', 'PD-L1': 'PD-L1', 'Beta-catenin': 'Beta-catenin', 'CD3E': 'CD3E', 'CD45RO': 'CD45RO', 'GranzymeB': 'GranzymeB',
        'alphaSMA': 'alphaSMA', 'K8_18': 'PanCK'}
markers = {}
for c, a in warped.items():
    cells[f'truth_{c}'] = np.bincount(lab.ravel(), weights=a.ravel().astype(np.float64), minlength=n)[1:] / np.maximum(area[1:], 1)
    if c in PROT:
        p = PROT[c]
        markers[c] = dict(truth=f'truth_{c}', components=[p], own=OWN.get(p, []))
with h5py.File(f'{XD}/{sample}_cell_feature_matrix.h5', 'r') as f:
    m = f['matrix']
    bc = np.array([b.decode() for b in m['barcodes'][:]]); ft = np.array([t.decode() for t in m['features']['feature_type'][:]])
    fname = np.array([t.decode() for t in m['features']['name'][:]])
    X = sp.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=tuple(m['shape'][:])).T.tocsr()
X = X[:, ft == 'Gene Expression']; fname = fname[ft == 'Gene Expression']
row = pd.Series(np.arange(len(bc)), index=bc).reindex(cell_ids)
Xe = sp.vstack([X, sp.csr_matrix((1, X.shape[1]), dtype=X.dtype)]).tocsr()
Xc = Xe[np.where(row.notna(), row.fillna(-1).values, X.shape[0]).astype(np.int64)].astype(np.float32)
sp.save_npz(f'{cd}/counts.npz', Xc); json.dump(list(fname), open(f'{cd}/counts_genes.json', 'w'))
cells['total_counts'] = np.asarray(Xc.sum(1)).ravel()
cells.to_parquet(f'{cd}/cells.parquet')
meta = dict(dataset=name, platform='xenium', tissue='CRLM' if 'Normal' not in sample else 'liver_adjacent_normal', panel='custom462',
            protein_platform='IMC', um_per_px=UM, H=hc, W=wc, crop=[y0, y1, x0, x1], n_cells=n - 1, n_cells_in_roi=int(inroi.sum()),
            vocab_genes_present=len(genes), coarse_ncc=float(score), ecc_cc=float(cc), reg_dna_dapi_r=float(r_reg),
            affine_imc_px_to_crop_px=M[:2].tolist(), roi=meta_roi, eval_mask='in_roi', markers=markers)
json.dump(meta, open(f'{cd}/meta.json', 'w'), indent=1)
np.save(f'{cd}/reg_check.npy', np.stack([Dc, norm(dn)]).astype(np.float16))
log('saved', cd, 'cells', n - 1, 'in ROI', int(inroi.sum()))
