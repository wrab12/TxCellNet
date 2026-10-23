"""Image-level comparison on the kidney held-out stripes, where the measured 27-plex image is available.

Methods: TxCellNet stage 1 (conditional mean), TxCellNet stage 2 (refined, constrained), cell-level baselines painted
into their cells (NicheTrans, MLP; per-cell predictions from baselines_cell.py), cognate-mRNA density (Gaussian
sigma 2 px). Painted and mRNA images are put on the image scale by an affine map per protein fitted on training-stripe
pixels (painted: a*p + b inside cells and a constant outside), so that only the spatial pattern is compared.
Metrics per protein, held-out stripes:
  pix_r     Pearson over tissue pixels
  within_r  Pearson of (pixel - its cell mean) over in-cell pixels: sub-cellular pattern (0 for painted predictions)
  enrich    median over cells of (mean on the cell-boundary ring - mean in the interior), z units: membrane vs inner
            localization; reported for the measured image and each method
NicheTrans: trained as in its official tutorial (nichetrans_official.py; panel-matched kidney model, seed 1).
usage: [ONLY_PAINT=1] image_metrics.py MEAN_RUN REFINE_RUN OUT_CSV   (ONLY_PAINT=1: NicheTrans rows only)
"""
import json, os, sys
import numpy as np, pandas as pd, cv2
from scipy.ndimage import gaussian_filter

MEAN, REF, OUT = sys.argv[1:4]
SD = ['data/xenium_protein_rcc/pilot_L2', 'data/xenium_protein_ccrcc/pilot_L2']; BT = ['rcc', 'ccrcc']; CASE = ['kidney_rcc', 'kidney_ccrcc']
prots = json.load(open(f'{SD[0]}/meta.json'))['prots']; OWN = json.load(open(f'{SD[0]}/meta.json'))['own_genes']; NP = len(prots)
ONLY = os.environ.get('ONLY_PAINT') == '1'
BASE = {'NicheTrans (painted)': 'results/bench/nt_off_full_s1/preds/{case}__nichetrans_off.npy'}
if not ONLY: BASE['MLP (painted)'] = 'results/bench/cellbase_s1/preds/{case}__mlp_pm.npy'
rows = []
for sd, bt, case in zip(SD, BT, CASE):
    meta = json.load(open(f'{sd}/meta.json'))
    valid, tissue = np.load(f'{sd}/valid.npy'), np.load(f'{sd}/tissue.npy'); H, W = valid.shape
    edges = np.array(json.load(open(f'results/cell_baseline/{bt}_stripes.json'))['x_edges_um']) / meta['um_per_px']
    tcol = (np.digitize(np.arange(W), edges) % 2 == 1)[None, :]
    inside = valid & tissue; test, fit = inside & tcol, inside & ~tcol
    ii = np.flatnonzero(fit.ravel())[np.random.default_rng(0).choice(int(fit.sum()), 2_000_000)]
    lab = np.load(f'{sd}/cells.npy'); labr = lab.ravel(); n = int(labr.max()) + 1; incell = labr > 0
    cnt = np.bincount(labr, minlength=n).astype(np.float64)
    b = np.zeros_like(lab, bool)
    b[:-1] |= lab[:-1] != lab[1:]; b[1:] |= lab[1:] != lab[:-1]; b[:, :-1] |= lab[:, :-1] != lab[:, 1:]; b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b &= lab > 0
    dist = cv2.distanceTransform(((lab > 0) & ~b).astype(np.uint8), cv2.DIST_L2, 3)
    ring, core = b.ravel(), ((dist >= 2) & (lab > 0)).ravel()
    nring = np.bincount(labr[ring], minlength=n); ncore = np.bincount(labr[core], minlength=n)
    tcells = np.zeros(n, bool); tcells[np.unique(labr[test.ravel() & incell])] = True
    good = tcells & (nring >= 4) & (ncore >= 4); good[0] = False
    rng = np.random.default_rng(0)
    pix = rng.choice(np.flatnonzero(test.ravel()), 2_000_000, replace=False); pin = pix[incell[pix]]
    fpix = rng.choice(np.flatnonzero(fit.ravel()), 2_000_000, replace=False)
    # per-cell baseline predictions -> pilot labels via cell ids
    ids = pd.read_csv(f'{sd}/cell_ids.csv')['cell_id'].astype(str).values
    pos = pd.Series(np.arange(1, len(ids) + 1), index=ids)
    cells = pd.read_parquet(f'data/cases/{case}/cells.parquet')
    lab_of_row = pos.reindex(cells['cell_id'].astype(str).values).values
    okr = ~np.isnan(lab_of_row); lab_of_row = lab_of_row[okr].astype(int)
    T = np.load(f'{sd}/tx.npz'); genes = meta['genes']
    mu_img = np.load(f'{sd}/runs/{MEAN}/pred_full.npy', mmap_mode='r'); ref_img = np.load(f'{sd}/runs/{REF}/refined_noise0.85.npy', mmap_mode='r')
    Yr = np.load(f'{sd}/Y.npy', mmap_mode='r')
    bp = {k: np.load(v.format(case=case)).astype(np.float32) for k, v in BASE.items()}

    def enrich(flat):
        rs = np.bincount(labr[ring], weights=flat[ring], minlength=n) / np.maximum(nring, 1)
        cs = np.bincount(labr[core], weights=flat[core], minlength=n) / np.maximum(ncore, 1)
        return float(np.median((rs - cs)[good]))

    def score(flat, yflat, ycm, method, j, e_true):
        cm = np.bincount(labr, weights=flat, minlength=n) / np.maximum(cnt, 1)
        wp = flat[pin] - cm[labr[pin]]; wt = yflat[pin] - ycm[labr[pin]]
        wr = float(np.corrcoef(wp, wt)[0, 1]) if wp.std() > 1e-6 else 0.0
        return dict(section=bt, protein=prots[j], method=method, pix_r=float(np.corrcoef(flat[pix], yflat[pix])[0, 1]),
                    within_r=wr, enrich=enrich(flat), enrich_measured=e_true)

    for j, p in enumerate(prots):
        a = np.asarray(Yr[meta['prots'].index(p)], np.float32); v = a.ravel()[ii]
        y = ((a - v.mean()) / (v.std() + 1e-4)).ravel().astype(np.float64)
        ycm = np.bincount(labr, weights=y, minlength=n) / np.maximum(cnt, 1)
        e_true = enrich(y)
        if not ONLY:
            rows.append(dict(section=bt, protein=p, method='measured', pix_r=1.0, within_r=1.0, enrich=e_true, enrich_measured=e_true))
            rows.append(score(np.asarray(mu_img[j], np.float64).ravel(), y, ycm, 'TxCellNet stage 1 (mean)', j, e_true))
            rows.append(score(np.asarray(ref_img[j], np.float64).ravel(), y, ycm, 'TxCellNet stage 2 (refined)', j, e_true))
        for k, P_ in bp.items():  # painted: a*p + b inside cells, c outside (fitted on training-stripe pixels)
            lut = np.full(n, np.nan); lut[lab_of_row] = P_[okr, j]
            pv = lut[labr]; m_in = ~np.isnan(pv)
            f_in = fpix[m_in[fpix]]; f_out = fpix[~m_in[fpix]]
            A_, B_ = np.polyfit(pv[f_in], y[f_in], 1); c_out = y[f_out].mean() if len(f_out) else 0.0
            flat = np.where(m_in, A_ * np.nan_to_num(pv) + B_, c_out)
            rows.append(score(flat, y, ycm, k, j, e_true))
        gs = [genes.index(g) for g in OWN.get(p, []) if g in genes]
        if gs and not ONLY:
            k = np.isin(T['gid'], gs); d = np.zeros((H, W), np.float32)
            np.add.at(d, (T['gy'][k].astype(np.int64), T['gx'][k].astype(np.int64)), 1)
            flat = np.log1p(gaussian_filter(d, 2.0) * 10).ravel().astype(np.float64)
            A_, B_ = np.polyfit(flat[fpix], y[fpix], 1)
            rows.append(score(A_ * flat + B_, y, ycm, 'cognate mRNA (density)', j, e_true))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(bt, p, flush=True)
D = pd.DataFrame(rows)
print(D.groupby('method')[['pix_r', 'within_r']].median().round(3).to_string())
for m, g in D[D.method != 'measured'].groupby('method'):
    print(m, 'enrichment agreement across proteins (r):', round(float(np.corrcoef(g.enrich, g.enrich_measured)[0, 1]), 3) if g.enrich.std() > 1e-6 else 0)
