"""Case study: tumour-immune architecture of colorectal liver metastases from RNA alone, checked against same-section IMC.

Sections: the five tumour ROIs with same-section IMC (tumour 1, tumour 2, tumour 3 A1-A3); the adjacent-normal section
contains almost no tumour and is not analysed. All models are trained on kidney only.
Sources of per-cell protein values (the cells scored in the main benchmark): IMC (measured), TxCellNet (six-network
ensemble), NicheTrans (three-seed ensemble), DGAT, and the cognate mRNA (log1p of the coding-gene counts per 1,000 counts).
The same rule is applied to every source:
  positive cells   value above the Otsu threshold of the section (tumour PanCK, immune CD45, macrophage CD68, T cell CD3E)
  tumour region    cells whose neighbourhood (radius R) is at least half tumour cells
  signed distance  of each cell to the region boundary (negative inside the tumour region)
  profiles         fraction of immune / macrophage / T cells per 20-um distance bin (-150 to +150 um, bins with >= 20 cells);
                   95% intervals from a spatial block bootstrap (200-um tiles resampled with replacement, 2,000 draws)
  exclusion index  immune fraction within 60 um inside the boundary divided by that within 60 um outside
Agreement with IMC: Dice of tumour regions and of immune cells, Pearson r of the profiles across bins.
usage: crlm_case.py OUT_DIR  -> OUT_DIR/{summary.csv, profiles.csv, thresholds.csv, map_<case>.npz}
"""
import json, os, sys, glob
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu

from txcellnet.bench import PROTS

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
CASES = ['crlm_tumor1', 'crlm_tumor2', 'crlm_tumor3_A1', 'crlm_tumor3_A2', 'crlm_tumor3_A3']
GROUP = {'crlm_tumor1': 'crlm_462', 'crlm_tumor2': 'crlm_462',
         'crlm_tumor3_A1': 'crlm_319', 'crlm_tumor3_A2': 'crlm_319', 'crlm_tumor3_A3': 'crlm_319'}
SOURCES = ['IMC', 'TxCellNet', 'NicheTrans', 'DGAT', 'Cognate mRNA']
ROLE = {'tumour': 'PanCK', 'immune': 'CD45', 'macrophage': 'CD68', 'tcell': 'CD3E'}
R, MARGIN, TILE, BOOT = 25.0, 60.0, 200.0, 2000
BINS = np.arange(-150, 151, 20.0); MID = (BINS[:-1] + BINS[1:]) / 2


def load(c):
    cd = f'data/cases/{c}'; meta = json.load(open(f'{cd}/meta.json')); cells = pd.read_parquet(f'{cd}/cells.parquet')
    X = sp.load_npz(f'{cd}/counts.npz').tocsr(); genes = json.load(open(f'{cd}/counts_genes.json'))
    ok = (cells.area_px.values >= 10) & (np.asarray(X.sum(1)).ravel() >= 1) & cells[meta['eval_mask']].values.astype(bool)
    idx = np.flatnonzero(ok); xy = cells[['x_um', 'y_um']].values[idx]; src = {s: {} for s in SOURCES}
    for k in ROLE.values():
        mk = meta['markers'][k]; t = cells[mk['truth']].values.astype(float)
        src['IMC'][k] = np.log1p(np.clip(t - mk.get('offset', 0), 0, None))[idx]
    txs = sorted(glob.glob(f'results/bench/predcache/{c}__fin_*_s[123].npy')); assert len(txs) == 6, txs
    nts = sorted(glob.glob(f'results/bench/nt_off_{GROUP[c]}_s[123]/preds/{c}__nichetrans_off*.npy')); assert len(nts) == 3, nts
    pred = {'TxCellNet': np.mean([np.load(f).astype(float) for f in txs], 0),
            'NicheTrans': np.mean([np.load(f).astype(float) for f in nts], 0),
            'DGAT': np.load(f'results/bench/dgat_{GROUP[c]}/preds/{c}__dgat_pm.npy').astype(float)}
    for s, P in pred.items():
        src[s] = {k: P[idx, PROTS.index(k)] for k in ROLE.values()}
    tot = np.asarray(X.sum(1)).ravel()[idx]; gi = {g: i for i, g in enumerate(genes)}; Xi = X[idx]
    for k in ROLE.values():
        own = [gi[g] for g in meta['markers'][k].get('own', []) if g in gi]; assert own, (c, k)
        src['Cognate mRNA'][k] = np.log1p(1e3 * np.asarray(Xi[:, own].sum(1)).ravel() / np.maximum(tot, 1))
    return dict(xy=xy, src=src, depth=float(np.median(tot)))


def region(xy, pos, tree):
    return np.array([pos[n].mean() for n in tree.query_ball_point(xy, R)]) >= 0.5


def signed_distance(xy, reg):
    if reg.all() or (~reg).all(): return np.full(len(xy), np.nan)
    return np.where(reg, -cKDTree(xy[~reg]).query(xy)[0], cKDTree(xy[reg]).query(xy)[0])


def profile(sd, flag):
    b = np.digitize(sd, BINS); n = np.array([(b == i).sum() for i in range(1, len(BINS))])
    return np.array([flag[b == i].mean() if n[i - 1] >= 20 else np.nan for i in range(1, len(BINS))]), n


def boot_profile(xy, sd, flag, rng):
    """95% interval of the binned fraction, resampling 200-um tiles with replacement."""
    tile = np.unique(np.floor(xy / TILE).astype(int), axis=0, return_inverse=True)[1].ravel(); b = np.digitize(sd, BINS) - 1
    ok = (b >= 0) & (b < len(MID)); T = tile.max() + 1
    N = np.zeros((T, len(MID))); K = np.zeros((T, len(MID)))
    np.add.at(N, (tile[ok], b[ok]), 1); np.add.at(K, (tile[ok], b[ok]), flag[ok])
    W = rng.multinomial(T, np.full(T, 1 / T), size=BOOT).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'): F = (W @ K) / (W @ N)
    return np.nanpercentile(F, 2.5, 0), np.nanpercentile(F, 97.5, 0)


def corr(a, b):
    f = np.isfinite(a) & np.isfinite(b); return np.corrcoef(a[f], b[f])[0, 1] if f.sum() > 3 else np.nan


D = {c: load(c) for c in CASES}; print('loaded', flush=True)
thr = {}
for s in SOURCES:
    for k in ROLE.values():
        for c in CASES:
            thr[(c, s, k)] = threshold_otsu(D[c]['src'][s][k])
pd.DataFrame([dict(case=c, source=s, marker=k, threshold=v) for (c, s, k), v in thr.items()]).to_csv(f'{OUT}/thresholds.csv', index=False)

rows, prof_rows = [], []; rng = np.random.default_rng(0)
for c in CASES:
    xy = D[c]['xy']; tree = cKDTree(xy); ref = None; maps = {}
    for s in SOURCES:
        pos = {r: D[c]['src'][s][k] > thr[(c, s, k)] for r, k in ROLE.items()}
        reg = region(xy, pos['tumour'], tree); sd = signed_distance(xy, reg)
        prof = {r: profile(sd, pos[r]) for r in ('immune', 'macrophage', 'tcell')}
        ci = {r: boot_profile(xy, sd, pos[r], rng) for r in ('immune', 'macrophage')}
        if s == 'IMC': ref = dict(reg=reg, imm=pos['immune'], prof={r: v[0] for r, v in prof.items()})
        inner, outer = (sd < 0) & (sd >= -MARGIN), (sd >= 0) & (sd < MARGIN)
        rows.append(dict(case=c, source=s, cells=len(xy), median_counts=D[c]['depth'],
                         tumour_cells=pos['tumour'].mean(), tumour_region=reg.mean(), immune=pos['immune'].mean(),
                         dice_tumour=2 * (reg & ref['reg']).sum() / max(reg.sum() + ref['reg'].sum(), 1),
                         dice_immune=2 * (pos['immune'] & ref['imm']).sum() / max(pos['immune'].sum() + ref['imm'].sum(), 1),
                         r_immune=corr(prof['immune'][0], ref['prof']['immune']),
                         r_macrophage=corr(prof['macrophage'][0], ref['prof']['macrophage']),
                         r_tcell=corr(prof['tcell'][0], ref['prof']['tcell']),
                         immune_in=pos['immune'][inner].mean() if inner.any() else np.nan,
                         immune_out=pos['immune'][outer].mean() if outer.any() else np.nan))
        prof_rows += [dict(case=c, source=s, distance=m, n=prof['immune'][1][i], **{r: prof[r][0][i] for r in prof},
                           **{f'{r}_{q}': ci[r][j][i] for r in ci for j, q in enumerate(('lo', 'hi'))}) for i, m in enumerate(MID)]
        maps[s.replace(' ', '_')] = np.stack([pos['tumour'], pos['immune'], reg]).astype(np.uint8)
    np.savez_compressed(f'{OUT}/map_{c}.npz', xy=xy.astype(np.float32), **maps)
    print(c, 'done', flush=True)
S = pd.DataFrame(rows); S['exclusion'] = S.immune_in / S.immune_out
S.to_csv(f'{OUT}/summary.csv', index=False); pd.DataFrame(prof_rows).to_csv(f'{OUT}/profiles.csv', index=False)
pd.set_option('display.width', 250); print(S.drop(columns=['cells']).round(3).to_string())
