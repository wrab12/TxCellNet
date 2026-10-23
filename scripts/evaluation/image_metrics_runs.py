"""Image-level metrics of several stage-1 mean images on the kidney held-out stripes (ablations): pixel Pearson,
within-cell Pearson (pixel minus its cell mean) and membrane-vs-interior enrichment agreement, as in image_metrics.py.
usage: image_metrics_runs.py OUT_CSV RUN[,RUN...]     (each RUN has pilot_L2/runs/RUN/pred_full.npy for both sections)
"""
import json, sys
import numpy as np, pandas as pd, cv2

OUT, RUNS = sys.argv[1], sys.argv[2].split(',')
SD = ['data/xenium_protein_rcc/pilot_L2', 'data/xenium_protein_ccrcc/pilot_L2']; BT = ['rcc', 'ccrcc']
prots = json.load(open(f'{SD[0]}/meta.json'))['prots']
rows = []
for sd, bt in zip(SD, BT):
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
    Yr = np.load(f'{sd}/Y.npy', mmap_mode='r')
    imgs = {r: np.load(f'{sd}/runs/{r}/pred_full.npy', mmap_mode='r') for r in RUNS}

    def enrich(flat):
        rs = np.bincount(labr[ring], weights=flat[ring], minlength=n) / np.maximum(nring, 1)
        cs = np.bincount(labr[core], weights=flat[core], minlength=n) / np.maximum(ncore, 1)
        return float(np.median((rs - cs)[good]))

    for j, p in enumerate(prots):
        a = np.asarray(Yr[meta['prots'].index(p)], np.float32); v = a.ravel()[ii]
        y = ((a - v.mean()) / (v.std() + 1e-4)).ravel().astype(np.float64)
        ycm = np.bincount(labr, weights=y, minlength=n) / np.maximum(cnt, 1); e_true = enrich(y)
        for r in RUNS:
            flat = np.asarray(imgs[r][j], np.float64).ravel()
            cm = np.bincount(labr, weights=flat, minlength=n) / np.maximum(cnt, 1)
            rows.append(dict(section=bt, protein=p, run=r, pix_r=float(np.corrcoef(flat[pix], y[pix])[0, 1]),
                             within_r=float(np.corrcoef(flat[pin] - cm[labr[pin]], y[pin] - ycm[labr[pin]])[0, 1]),
                             enrich=enrich(flat), enrich_measured=e_true))
        pd.DataFrame(rows).to_csv(OUT, index=False); print(bt, p, flush=True)
D = pd.DataFrame(rows)
print(D.groupby('run')[['pix_r', 'within_r']].median().round(3).to_string())
for r, g in D.groupby('run'): print(r, 'enrichment agreement r =', round(float(np.corrcoef(g.enrich, g.enrich_measured)[0, 1]), 3))
