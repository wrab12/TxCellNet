"""Why stage 2: is the refined image (stage 2) closer to the measured image than the conditional mean (stage 1)?

Kidney held-out stripes (RCC, ccRCC; 27 IF proteins), the exact images of the paper:
  measured  per-section z of log1p(max(protein - 100, 0)) (Y.npy; z parameters from the 2M training-stripe pixels of
            scripts/train/train_refine.py, values rounded to float16 as there)
  stage 1   runs/fin_ens6/pred_full.npy (6-network ensemble mean mu-hat)
  stage 2   runs/refine_fin/refined_noise0.85.npy (scripts/train/train_refine.py: tau 0.85, 20 Euler steps, noise seed 1000, one sample,
            per-cell mean constraint)
  controls  stage 1 + matched random noise (matched_noise): Gaussian noise whose SD over the held-out pixels equals the SD
            of the stage-2 residual (stage 2 - stage 1), with each cell's noise mean removed so that cell means stay those
            of stage 1, as in stage 2. 'noise' = i.i.d. per pixel; 'noise_s1' = Gaussian-smoothed (sigma 1 px), rescaled.
            Cell selections and window / patch locations are those of the three main images, so adding the controls leaves
            every stage-1 / stage-2 number unchanged.
Held-out region = valid & tissue & held-out stripe columns; pixel sample (2M) and the 48 windows of 256 x 256 px are drawn
exactly as in scripts/train/train_refine.py, so pix_r / within_r / sharp / hfpow / sd / W1 reproduce runs/refine_fin/metrics.csv.

Metric families (per protein and section; each compares stage 1 and stage 2 with the measured image):
  (1) pixel intensity distribution: W1 (1-Wasserstein, z units) on the 2M held-out pixels
  (2) Haralick / GLCM texture (skimage graycomatrix/graycoprops: contrast, homogeneity, energy, correlation) on the 48
      windows; grey levels: 32 equal-width bins between the measured image's 0.5th and 99.5th held-out percentiles, the
      same bins for all three images (as CellProfiler MeasureTexture, which quantizes every image on one fixed scale);
      distances 1 and 3 px (3 = CellProfiler default), 4 angles averaged; feature = mean over windows; reported as the
      ratio to the measured value. (Quantile binning per image is not used: the measured image has a point mass at 0
      after background subtraction, so equal-frequency bins cannot be built for it.)
  (3) CellProfiler-style per-cell features on the linear background-subtracted intensity expm1(log1p scale), held-out
      cells (every pixel valid, in tissue and in a held-out stripe; >= 20 px):
        sd          within-cell intensity SD               (MeasureObjectIntensity StdIntensity)
        cv          SD / mean                              (cells with a positive mean in all three images)
        edge_centre mean on the boundary ring / mean of the interior (distance to boundary >= 2 px)
                    (MeanIntensityEdge / interior mean; cells with >= 4 px in both and a positive interior mean)
        radial_edge, radial_centre  MeanFrac of the outer and inner quarter of normalized distance to the boundary
                    (MeasureObjectIntensityDistribution, 4 bins)
        gs1..gs3    granular spectrum (MeasureGranularity): fraction of the cell's intensity removed between grey openings
                    with disks of radius k-1 and k px (k = 1, 2, 3), here per cell and without the background step;
        gran        gs1 + gs2 + gs3 = fraction removed by the radius-3 opening
        sd_log      within-cell SD on the log1p scale (robustness of 'sd' to the intensity scale)
      distance to measured = two-sample KS statistic between the per-cell feature distributions (also W1 / measured IQR)
  (4) image-distribution distance without pretrained weights: sliced Wasserstein distance between 7 x 7 Laplacian-pyramid
      patch distributions (the SWD of Karras et al., ICLR 2018), 5 levels (256 ... 16 px), 128 patches per window and level,
      descriptors normalized by the measured set's mean and SD, 4 x 128 random directions; x 1e3 as in that paper.
      (A deep-feature distance, KID on Cellpose features, is in stage2_kid_cellpose.py.)
  (5) trade-off: pixel Pearson and within-cell Pearson (as in the paper)
Summary (medians, Wilcoxon): stage2_realism_summary.py.
usage: [WORKERS=9] [ONLY=0,5] [SECTIONS=rcc,ccrcc] stage2_realism.py OUT_DIR   (run from the project root)
"""
import json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from scipy.stats import ks_2samp, wasserstein_distance
from skimage.feature import graycomatrix, graycoprops

cv2.setNumThreads(2)
SD ={'rcc': 'data/xenium_protein_rcc/pilot_L2', 'ccrcc': 'data/xenium_protein_ccrcc/pilot_L2'}
MEAN_RUN, REF_RUN = 'fin_ens6', 'refine_fin'
prots = json.load(open(f'{SD["rcc"]}/meta.json'))['prots']; NP = len(prots)
GLCM_PROPS = ['contrast', 'homogeneity', 'energy', 'correlation']; GLCM_D = [1, 3]; NLEV = 32
ANG = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
CELLF = ['sd', 'cv', 'edge_centre', 'radial_edge', 'radial_centre', 'gran', 'gs1', 'gs2', 'gs3', 'sd_log']
IMGS = ['measured', 'stage1', 'stage2']               # images that define the cell selections
CTRL = {'noise': 0.0, 'noise_s1': 1.0}                 # matched-noise controls: Gaussian smoothing sigma (px), 0 = i.i.d.
COMP = ['stage1', 'stage2'] + list(CTRL)               # images compared with the measured one
ALL = IMGS + list(CTRL)
SECIDX = {'rcc': 0, 'ccrcc': 1}
fy =np.fft.fftfreq(256)[:, None]; fx = np.fft.fftfreq(256)[None, :]; HF = np.sqrt(fy ** 2 + fx ** 2) > 0.2
G = {}  # section-level arrays, set in the parent before forking


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def setup(bt):
    sd = SD[bt]; meta = json.load(open(f'{sd}/meta.json'))
    valid, tissue = np.load(f'{sd}/valid.npy'), np.load(f'{sd}/tissue.npy'); H, W = valid.shape
    edges = np.array(json.load(open(f'results/cell_baseline/{bt}_stripes.json'))['x_edges_um']) / meta['um_per_px']
    test_col = np.digitize(np.arange(W), edges) % 2 == 1
    inside = valid & tissue; fit, test = inside & ~test_col[None, :], inside & test_col[None, :]
    ii = np.flatnonzero(fit.ravel())[np.random.default_rng(0).choice(int(fit.sum()), 2_000_000)]  # as scripts/train/train_refine.py
    lab = np.load(f'{sd}/cells.npy'); labr = lab.ravel(); n = int(labr.max()) + 1; incell = labr > 0
    cnt = np.bincount(labr, minlength=n).astype(np.float64)
    rng2 = np.random.default_rng(0); pix = rng2.choice(np.flatnonzero(test.ravel()), 2_000_000, replace=False)
    pin = pix[incell[pix]]
    crops = []
    while len(crops) < 48:
        y0, x0 = int(rng2.integers(0, H - 256)), int(rng2.integers(0, W - 256))
        if test[y0:y0 + 256, x0:x0 + 256].mean() > 0.95: crops.append((y0, x0))
    # cell geometry (as train_refine.py geometry / image_metrics.py)
    b = np.zeros_like(lab, bool)
    b[:-1] |= lab[:-1] != lab[1:]; b[1:] |= lab[1:] != lab[:-1]; b[:, :-1] |= lab[:, :-1] != lab[:, 1:]; b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b &= lab > 0
    dist = cv2.distanceTransform(((lab > 0) & ~b).astype(np.uint8), cv2.DIST_L2, 3).ravel()
    # held-out cells: every pixel valid, in tissue and in a held-out stripe; >= 20 px
    ntest = np.bincount(labr, weights=test.ravel().astype(np.float64), minlength=n)
    hc = (ntest == cnt) & (cnt >= 20); hc[0] = False
    P = np.flatnonzero(hc[labr])                       # pixels of held-out cells
    cid = np.cumsum(hc) - 1                            # compact cell index
    L = cid[labr[P]]; nc = int(hc.sum())
    ring = b.ravel()[P]; core = dist[P] >= 2
    dmax = pd.Series(dist[P]).groupby(L).max().reindex(range(nc)).fillna(0).values
    rel = dist[P] / np.maximum(dmax[L], 1e-6)
    rbin = np.minimum((rel * 4).astype(np.int8), 3)  # 0 = outer quarter (edge) ... 3 = inner quarter (centre)
    G.update(bt=bt, sd=sd, meta=meta, H=H, W=W, ii=ii, labr=labr, n=n, cnt=cnt, pix=pix, pin=pin, crops=crops, P=P, L=L, nc=nc,
             ring=ring, core=core, rbin=rbin, cnt_c=np.bincount(L, minlength=nc).astype(np.float64),
             nring=np.bincount(L[ring], minlength=nc), ncore=np.bincount(L[core], minlength=nc),
             nbin=np.stack([np.bincount(L[rbin == k], minlength=nc) for k in range(4)]).astype(np.float64))
    log(bt, f'H {H} W {W}; held-out pixels {int(test.sum())}; held-out cells {nc}; windows {len(crops)}')


def matched_noise(mu, s2, labr, cnt, pix, bt, j, smooth):
    """control: stage 1 + Gaussian noise with the SD of the stage-2 residual over the held-out pixels `pix`, each cell's
    noise mean removed (stage-1 cell means kept, as in stage 2); smooth = Gaussian sigma in px (0 = i.i.d.).
    Deterministic in (section, protein, smooth); also used by stage2_kid_cellpose.py."""
    sig = float((s2.ravel()[pix] - mu.ravel()[pix]).std())
    e = np.random.default_rng([7, SECIDX[bt], j, int(smooth > 0)]).standard_normal(mu.shape, dtype=np.float32)
    if smooth > 0: e = cv2.GaussianBlur(e, (0, 0), smooth)
    er = e.ravel(); cm = np.bincount(labr, weights=er.astype(np.float64), minlength=len(cnt)) / np.maximum(cnt, 1); cm[0] = 0
    er = er - cm[labr].astype(np.float32)
    er *= np.float32(sig / float(er[pix].std()))
    return (mu + er.reshape(mu.shape)).astype(np.float32), sig


def win_stats(img):
    """sharpness and high-frequency power over the 48 windows (train_refine.py stats)"""
    cr = G['crops']
    g = np.mean([(np.abs(np.diff(img[y:y + 256, x:x + 256], axis=0))[:, :-1] + np.abs(np.diff(img[y:y + 256, x:x + 256], axis=1))[:-1, :]).mean() for y, x in cr])
    pw = np.mean([(np.abs(np.fft.fft2(img[y:y + 256, x:x + 256] - img[y:y + 256, x:x + 256].mean())) ** 2)[HF].sum() for y, x in cr])
    return g, pw


def glcm(q):
    """mean over windows of the 4 Haralick features at each distance; q = integer image in [0, NLEV)"""
    acc = {(p, d): [] for p in GLCM_PROPS for d in GLCM_D}
    for y, x in G['crops']:
        M = graycomatrix(q[y:y + 256, x:x + 256], distances=GLCM_D, angles=ANG, levels=NLEV, symmetric=True, normed=True)
        for p in GLCM_PROPS:
            v = graycoprops(M, p)  # (distances, angles)
            for k, d in enumerate(GLCM_D): acc[(p, d)].append(v[k].mean())
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def lap_pyramid(a, levels=5):
    out, g = [], a
    for _ in range(levels - 1):
        d = cv2.pyrDown(g); out.append(g - cv2.pyrUp(d, dstsize=(g.shape[1], g.shape[0]))); g = d
    out.append(g)
    return out


def swd_desc(img, rng):
    """7x7 Laplacian-pyramid patches: list over levels of (n_patches, 49) arrays"""
    D = [[] for _ in range(5)]
    for y, x in G['crops']:
        pyr = lap_pyramid(np.ascontiguousarray(img[y:y + 256, x:x + 256], np.float32))
        for l, a in enumerate(pyr):
            h, w = a.shape; yy = rng.integers(0, h - 6, 128); xx = rng.integers(0, w - 6, 128)
            D[l].append(np.stack([a[i:i + 7, j:j + 7].ravel() for i, j in zip(yy, xx)]))
    return [np.concatenate(d) for d in D]


def swd(A, B, mu, sd, seed=0):
    rng = np.random.default_rng(seed); A = (A - mu) / sd; B = (B - mu) / sd; r = []
    for _ in range(4):
        dirs = rng.standard_normal((A.shape[1], 128)); dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
        pa, pb = np.sort(A @ dirs, 0), np.sort(B @ dirs, 0); r.append(np.abs(pa - pb).mean())
    return float(np.mean(r)) * 1e3


def cell_feats(lin, loglin):
    """per held-out cell CellProfiler-style features from a linear-intensity image (flat) and its log1p-scale image"""
    P, L, nc = G['P'], G['L'], G['nc']; v = lin[P]; c = G['cnt_c']
    s1 = np.bincount(L, v, nc); s2 = np.bincount(L, v * v, nc); m = s1 / c
    sd = np.sqrt(np.maximum(s2 / c - m * m, 0))
    lv = loglin[P]; lm = np.bincount(L, lv, nc) / c
    sd_log = np.sqrt(np.maximum(np.bincount(L, lv * lv, nc) / c - lm * lm, 0))
    mr = np.bincount(L[G['ring']], v[G['ring']], nc) / np.maximum(G['nring'], 1)
    mc = np.bincount(L[G['core']], v[G['core']], nc) / np.maximum(G['ncore'], 1)
    frac = np.stack([np.bincount(L[G['rbin'] == k], v[G['rbin'] == k], nc) for k in range(4)]) / np.maximum(s1, 1e-12)
    meanfrac = frac / np.maximum(G['nbin'] / c, 1e-12)
    return dict(mean=m, sd=sd, sd_log=sd_log, ring=mr, core=mc, radial_edge=meanfrac[0], radial_centre=meanfrac[3], sum=s1)


def granularity(lin2d):
    """per held-out cell intensity removed between grey openings with disks of radius k-1 and k px (k = 1, 2, 3)"""
    P, L, nc = G['P'], G['L'], G['nc']; prev = lin2d.ravel()[P]; out = []
    for r in (1, 2, 3):
        op = cv2.morphologyEx(lin2d, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).ravel()[P]
        out.append(np.bincount(L, prev - op, nc)); prev = op
    return out


def one(j):
    t0 = time.time(); bt, sd, meta = G['bt'], G['sd'], G['meta']; labr, n, cnt, pix, pin = G['labr'], G['n'], G['cnt'], G['pix'], G['pin']
    Yr = np.load(f'{sd}/Y.npy', mmap_mode='r'); a = np.asarray(Yr[meta['prots'].index(prots[j])], np.float32)
    v = a.ravel()[G['ii']]; zm, zs = float(v.mean()), float(v.std() + 1e-4)
    y = ((a - zm) / zs).astype(np.float16).astype(np.float32)   # exactly the stage-2 evaluation reference
    mu = np.asarray(np.load(f'{sd}/runs/{MEAN_RUN}/pred_full.npy', mmap_mode='r')[j], np.float32)
    s2 = np.asarray(np.load(f'{sd}/runs/{REF_RUN}/refined_noise0.85.npy', mmap_mode='r')[j], np.float32)
    imgs = dict(measured=y, stage1=mu, stage2=s2)
    sig = {}
    for k, sm in CTRL.items(): imgs[k], sig[k] = matched_noise(mu, s2, labr, cnt, pix, bt, j, sm)
    rows = []

    def add(metric, family, vals, dist=None, ref=None):
        r = dict(section=bt, protein=prots[j], metric=metric, family=family, measured=vals.get('measured', np.nan),
                 stage1=vals['stage1'], stage2=vals['stage2'])
        if dist is not None: r.update(dist_stage1=dist['stage1'], dist_stage2=dist['stage2'])
        for k in CTRL:
            r[k] = vals[k]
            if dist is not None: r[f'dist_{k}'] = dist[k]
        rows.append(r)

    # (5) correlations + (1) pixel distribution + window statistics (train_refine.py definitions)
    ycm = np.bincount(labr, weights=y.ravel().astype(np.float64), minlength=n) / np.maximum(cnt, 1)
    rpix = y.ravel()[pix]; rsort = np.sort(rpix); rwin = y.ravel()[pin] - ycm[labr[pin]]
    gref, pref = win_stats(y)
    st = {}
    for k in COMP:
        im = imgs[k]; cm = np.bincount(labr, weights=im.ravel().astype(np.float64), minlength=n) / np.maximum(cnt, 1)
        g, pw = win_stats(im); srt = np.sort(im.ravel()[pix])
        st[k] = dict(pix_r=np.corrcoef(im.ravel()[pix], rpix)[0, 1], within_r=np.corrcoef(im.ravel()[pin] - cm[labr[pin]], rwin)[0, 1],
                     sharp=g / gref, hfpow=pw / pref, sd=srt.std() / rsort.std(), W1=float(np.abs(srt - rsort).mean()))
    for m in ['pix_r', 'within_r']:
        add(m, 'correlation', {k: st[k][m] for k in st}, {k: 1 - st[k][m] for k in st})
    for m in ['sharp', 'hfpow', 'sd']:
        add(m, 'image statistic (ratio)', dict(measured=1.0, **{k: st[k][m] for k in st}), {k: abs(np.log(st[k][m])) for k in st})
    add('W1', 'pixel distribution', {k: st[k]['W1'] for k in st}, {k: st[k]['W1'] for k in st})
    add('noise_sd', 'control: noise SD (z units)', {k: sig.get(k, np.nan) for k in COMP})
    # (2) GLCM on the 48 windows
    lo, hi = np.percentile(rpix, [0.5, 99.5])
    fixed = {k: np.clip(((imgs[k] - lo) / (hi - lo) * NLEV).astype(np.int32), 0, NLEV - 1).astype(np.uint8) for k in ALL}
    for tag, Q in [('fixed', fixed)]:
        F = {k: glcm(Q[k]) for k in ALL}
        for (p, d) in F['measured']:
            ref = F['measured'][(p, d)]
            ratio = {k: F[k][(p, d)] / ref for k in COMP}
            add(f'glcm_{p}_d{d}_{tag}', 'GLCM texture (ratio)', dict(measured=ref, **{k: F[k][(p, d)] for k in COMP}),
                {k: abs(np.log(ratio[k])) if ratio[k] > 0 else np.nan for k in ratio})
            rows[-1].update({f'ratio_{k}': ratio[k] for k in COMP})
    # (3) per-cell CellProfiler-style features on the linear intensity scale
    CF, GR = {}, {}
    for k in ALL:
        logim = imgs[k] * zs + zm                                   # log1p(max(protein - 100, 0)) scale
        lin = np.expm1(np.maximum(logim, 0)).astype(np.float32)
        CF[k] = cell_feats(lin.ravel(), logim.ravel()); GR[k] = granularity(lin)
    ok_mean = np.all([CF[k]['mean'] > 0 for k in IMGS], 0)
    ok_ec = (G['nring'] >= 4) & (G['ncore'] >= 4) & np.all([CF[k]['core'] > 0 for k in IMGS], 0)
    feats = {}
    for k in ALL:
        c = CF[k]
        feats[k] = dict(sd=(c['sd'], np.ones(G['nc'], bool)), sd_log=(c['sd_log'], np.ones(G['nc'], bool)),
                        cv=(c['sd'] / np.maximum(c['mean'], 1e-12), ok_mean),
                        edge_centre=(c['ring'] / np.maximum(c['core'], 1e-12), ok_ec),
                        radial_edge=(c['radial_edge'], ok_mean), radial_centre=(c['radial_centre'], ok_mean),
                        gran=(sum(GR[k]) / np.maximum(c['sum'], 1e-12), ok_mean),
                        **{f'gs{r + 1}': (GR[k][r] / np.maximum(c['sum'], 1e-12), ok_mean) for r in range(3)})
    for f in CELLF:
        sel = feats['measured'][f][1]; ref = feats['measured'][f][0][sel]
        iqr = np.subtract(*np.percentile(ref, [75, 25])) + 1e-12
        vals = {k: float(np.median(feats[k][f][0][sel])) for k in ALL}
        ks = {k: ks_2samp(feats[k][f][0][sel], ref).statistic for k in COMP}
        w1 = {k: wasserstein_distance(feats[k][f][0][sel], ref) / iqr for k in COMP}
        add(f'cell_{f}', 'per-cell feature (KS)', vals, ks)
        rows[-1].update({f'w1iqr_{k}': w1[k] for k in COMP}, n_cells=int(sel.sum()))
    # (4) SWD on Laplacian-pyramid patches
    De = {k: swd_desc(imgs[k], np.random.default_rng(100 + j)) for k in ALL}    # same patch locations for all images
    lv = {k: [] for k in COMP}
    for l in range(5):
        mref, sref = De['measured'][l].mean(0), De['measured'][l].std(0) + 1e-6
        for k in lv: lv[k].append(swd(De[k][l], De['measured'][l], mref, sref, seed=l))
    for l in range(5):
        add(f'swd_L{l}', 'SWD (per level)', {k: lv[k][l] for k in lv}, {k: lv[k][l] for k in lv})
    add('swd', 'SWD (mean of levels)', {k: float(np.mean(lv[k])) for k in lv}, {k: float(np.mean(lv[k])) for k in lv})
    log(bt, prots[j], f'{time.time() - t0:.0f}s', 'W1', round(st['stage1']['W1'], 3), round(st['stage2']['W1'], 3),
        round(st['noise']['W1'], 3), 'swd', round(np.mean(lv['stage1']), 1), round(np.mean(lv['stage2']), 1), round(np.mean(lv['noise']), 1))
    return rows


if __name__ == '__main__':
    OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
    rows = []
    for bt in os.environ.get('SECTIONS', 'rcc,ccrcc').split(','):
        setup(bt)
        JJ = [int(v) for v in os.environ['ONLY'].split(',')] if os.environ.get('ONLY') else range(NP)  # ONLY: smoke tests
        with Pool(int(os.environ.get('WORKERS', 9))) as pool:
            for r in pool.imap_unordered(one, JJ): rows += r
        G.clear()
        pd.DataFrame(rows).to_csv(f'{OUT}/stage2_realism_per_protein.csv', index=False)
    D = pd.DataFrame(rows).sort_values(['metric', 'section', 'protein'])
    D.to_csv(f'{OUT}/stage2_realism_per_protein.csv', index=False)
    # check against the stored stage-2 evaluation (runs/refine_fin/metrics.csv)
    M = pd.read_csv(f'{SD["rcc"]}/runs/{REF_RUN}/metrics.csv')
    for m in ['pix_r', 'within_r', 'sharp', 'hfpow', 'sd', 'W1']:
        for k, (meth, cfg) in {'stage1': ('mean (stage 1)', '-'), 'stage2': ('refined (constrained)', 'noise0.85')}.items():
            ref = M[(M.method == meth) & (M.config == cfg)].set_index(['section', 'protein'])[m]
            mine = D[D.metric == m].set_index(['section', 'protein'])[k]
            dmax = float((mine - ref.reindex(mine.index)).abs().max())  # float16 storage of Y in train_refine.py: tiny differences
            log('check', m, k, 'max |diff| vs metrics.csv', f'{dmax:.2e}')
    log('done; summary: stage2_realism_summary.py')
