"""Fit, on the kidney Xenium Protein sections, how each multimodal-segmentation stain channel (one fluorescence channel
holding a cocktail of antibodies) relates to the individually imaged proteins of the same section.

For each kidney sample: per-cell means of the measured protein images in the model's target space (log1p(max(x-100,0)),
z-scored over tissue pixels) and per-cell log1p mean of the stain channels (level 2, 0.85 um/px). OLS of
stain ~ components gives the weights later applied to the *predicted* components on external sections.
Also reports how well the measured components explain the cocktail (Spearman of the fitted score), i.e. the
best a perfect component predictor could reach on that stain.

usage: fit_cocktail.py OUT_JSON
"""
import json, os, sys
import numpy as np, tifffile
from scipy.stats import spearmanr

from txcellnet.bench import PROTS, log

COCKTAILS = {'interior_aSMA_Vim': ('ch0003_alphasma_vimentin.ome.tif', ['alphaSMA', 'Vimentin']),
             'boundary_ATP1A1_Ecad_CD45': ('ch0001_atp1a1_cd45_e-cadherin.ome.tif', ['E-Cadherin', 'CD45'])}


def read_level(path, level=2):
    with tifffile.TiffFile(path, is_ome=False) as tf:
        return tf.series[0].levels[level].asarray()


out = {}
for s in ['rcc', 'ccrcc']:
    sd = f'data/xenium_protein_{s}/pilot_L2'
    Y = np.load(f'{sd}/Y.npy', mmap_mode='r'); lab = np.load(f'{sd}/cells.npy'); tissue = np.load(f'{sd}/tissue.npy')
    flat = lab.ravel(); n = int(lab.max()) + 1
    cnt = np.bincount(flat, minlength=n).astype(np.float64); ok = cnt > 0; ok[0] = False
    rng = np.random.default_rng(0); ii = np.flatnonzero(tissue.ravel()); ii = ii[rng.choice(len(ii), min(len(ii), 2_000_000), replace=False)]
    out[s] = {}
    for key, (fname, comps) in COCKTAILS.items():
        st = read_level(f'data/xenium_protein_{s}/outs/morphology_focus/{fname}').astype(np.float32)
        assert st.shape == lab.shape, (st.shape, lab.shape)
        t = np.log1p(np.bincount(flat, weights=st.ravel().astype(np.float64), minlength=n) / np.maximum(cnt, 1))[ok]
        Z = []
        for p in comps:
            a = np.asarray(Y[PROTS.index(p)], np.float32).ravel()
            mu, sdv = float(a[ii].mean()), float(a[ii].std()) + 1e-4
            Z.append((np.bincount(flat, weights=(a - mu) / sdv, minlength=n) / np.maximum(cnt, 1))[ok])
        Z = np.stack(Z, 1)
        A = np.c_[np.ones(len(t)), Z]
        coef = np.linalg.lstsq(A, t, rcond=None)[0]
        fit = A @ coef
        r2 = 1 - np.sum((t - fit) ** 2) / np.sum((t - t.mean()) ** 2)
        rho_fit = spearmanr(fit, t)[0]; rho_eq = spearmanr(Z.sum(1), t)[0]
        rho_each = {p: float(spearmanr(Z[:, i], t)[0]) for i, p in enumerate(comps)}
        out[s][key] = dict(components=comps, weights=[float(w) for w in coef[1:]], intercept=float(coef[0]), r2=float(r2),
                           spearman_fitted=float(rho_fit), spearman_equal=float(rho_eq), spearman_each=rho_each, n_cells=int(ok.sum()))
        log(s, key, 'weights', np.round(coef[1:], 3), 'R2', round(r2, 3), 'rho fitted', round(rho_fit, 3), 'equal', round(rho_eq, 3), rho_each)
for key in COCKTAILS:
    w = np.mean([out[s][key]['weights'] for s in ['rcc', 'ccrcc']], 0)
    out.setdefault('mean', {})[key] = dict(components=COCKTAILS[key][1], weights=[float(x) for x in w / np.abs(w).sum()])
os.makedirs(os.path.dirname(sys.argv[1]) or '.', exist_ok=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
log('saved', sys.argv[1])
