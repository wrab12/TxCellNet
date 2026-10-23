"""Robustness of the benchmark conclusions (Supplementary): other metrics, paired tests, single models, panel subsets.

For both tiers and every method: per-dataset median over markers of Spearman, Pearson and AUROC for Otsu-positive cells
(txcellnet.metrics), the mean over datasets, wins of TxCellNet, a one-sided Wilcoxon signed-rank test (TxCellNet > method,
paired by dataset) and a bootstrap 95% CI of the mean paired difference (10,000 resamples of datasets).
Within: predictions saved by train_case.py (cellpred.npy, columns = image channels), within_baselines.py and the published-
method scripts (runs/within_baselines/<method>.npy, columns = scored markers), scored on the held-out stripes; kidney rows
from the cross-dataset rows of the kidney held-out stripes (as in bench_tables.py). Cross: rows of results/bench/*/bench_rows.csv.
Also: single TxCellNet networks against single published models (cross), and the within tier without the 5K panels.
usage: robustness.py OUT_DIR
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import wilcoxon

from txcellnet.bench import load_case_cells as load_case, within_split, bench_rows
from txcellnet.metrics import metrics

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
METH = ['txcellnet', 'nichetrans', 'dgat', 'spamosaic', 'totalvi', 'mlp', 'ridge', 'knn', 'celltype', 'own_mRNA']
BREAST = lambda ss: [f'xen_breast_S{s}_{p}' for s in ss for p in ('Top', 'Mid', 'Bot')]
TEST = BREAST((2, 3, 4)) + ['xen_lymphnode_5k', 'crlm_tumor2', 'crlm_tumor3_A2', 'crlm_tumor3_A3', 'crlm_adjnormal', 'duchini_core02_non_tumour',
                            'duchini_core03_tonsil', 'duchini_core04_hca']
FRESH = ['xen_skin_5k', 'xen_prostate_5k', 'xen_ovary_5k', 'xen_ovary_v1ff', 'xen_breast_5k', 'xen_cervix_5k', 'xen_gbm_v1']
DEV = BREAST((1,)) + ['xen_lung_v1', 'xen_pancreas_v1', 'crlm_tumor1', 'crlm_tumor3_A1', 'duchini_core01_hcc']
EXT = DEV + TEST + FRESH
K5 = ['xen_lymphnode_5k', 'xen_skin_5k', 'xen_prostate_5k', 'xen_ovary_5k', 'xen_breast_5k', 'xen_cervix_5k', 'duchini_core01_hcc',
      'duchini_core02_non_tumour', 'duchini_core03_tonsil', 'duchini_core04_hca']
KMAP = {'txcellnet': 'fin_clean_s1', 'nichetrans': 'nichetrans_off_u', 'dgat': 'dgat_pm', 'spamosaic': 'spamosaic_pm', 'totalvi': 'totalvi_pm',
        'mlp': 'mlp_pm_u', 'ridge': 'ridge_pm_u', 'knn': 'knn_pm_u', 'celltype': 'celltype_pm_u', 'own_mRNA': 'own_mRNA'}
XMAP = {'txcellnet': 'txcellnet', 'nichetrans': 'nichetrans_off_e3', 'dgat': 'dgat_pm', 'spamosaic': 'spamosaic_pm', 'totalvi': 'totalvi_pm',
        'mlp': 'mlp_pm_e3', 'ridge': 'ridge_pm', 'knn': 'knn_pm', 'celltype': 'celltype_pm', 'own_mRNA': 'own_mRNA'}
WFILE = {'nichetrans': 'nichetrans_off'}  # NicheTrans trained as in its official tutorial (nichetrans_official.py)
SINGLE = ['fin_clean_s1', 'fin_clean_s2', 'fin_clean_s3', 'fin_rand_s1', 'fin_rand_s2', 'fin_rand_s3']
MET = ['spearman', 'pearson', 'auroc']

R = bench_rows()
CR = {m: R.groupby(['case', 'method'])[m].median().unstack('method') for m in MET}


def cross_table(metric, cmap, cases):
    T = CR[metric].reindex(cases)
    return pd.DataFrame({k: (T[v] if isinstance(v, str) and v in T else
                             T[[c for c in v if c in T]].mean(1) if not isinstance(v, str) else np.nan) for k, v in cmap.items()}, index=cases)


# ---------------- within tier: recompute all metrics from saved predictions ----------------
rows = []
for c in EXT:
    case = load_case(c); tr, te, keys, T, Rraw = within_split(case); cd = f'data/cases/{c}'
    yj = json.load(open(f'{cd}/Y.json')); chans = yj['channels']
    preds = {}
    p = f'{cd}/runs/within_clean/cellpred.npy'
    if os.path.exists(p):
        a = np.load(p); preds['txcellnet'] = np.stack([a[:, chans.index(yj['markers'][k])] for k in keys], 1)
    for m in ['nichetrans', 'dgat', 'spamosaic', 'totalvi', 'mlp', 'ridge', 'knn', 'celltype']:
        p = f'{cd}/runs/within_baselines/{WFILE.get(m, m)}.npy'
        if os.path.exists(p): preds[m] = np.load(p).astype(np.float64)
    X = case['X']; tot = np.asarray(X.sum(1)).ravel(); gi = {g: i for i, g in enumerate(case['genes'])}
    own = []
    for k in keys:
        idx = [gi[g] for g in case['meta']['markers'][k].get('own', []) if g in gi]
        own.append(np.log1p(np.asarray(X[:, idx].sum(1)).ravel() / np.maximum(tot, 1) * np.median(tot[tot > 0])) if idx else np.full(len(tot), np.nan))
    preds['own_mRNA'] = np.stack(own, 1)
    for m, P in preds.items():
        for j, k in enumerate(keys):
            if not np.isfinite(P[te, j]).any(): continue
            rows.append(dict(case=c, marker=k, method=m, **metrics(P[te, j], T[te, j])))
    print(c, sorted(preds), flush=True)
W = pd.DataFrame(rows); W.to_csv(f'{OUT}/within_rows_allmetrics.csv', index=False)
WT = {m: W.groupby(['case', 'method'])[m].median().unstack('method') for m in MET}
KID = ['kidney_rcc', 'kidney_ccrcc']
WITHIN = {m: pd.concat([cross_table(m, KMAP, KID), WT[m].reindex(EXT)[METH]]) for m in MET}
CROSS = {m: cross_table(m, XMAP, TEST + FRESH) for m in MET}


def compare(T):
    out = {}
    for k in METH[1:]:
        d = (T['txcellnet'] - T[k]).dropna()
        if len(d) < 3: continue
        bs = np.random.default_rng(0).choice(d.values, (10000, len(d))).mean(1)
        out[k] = dict(n=len(d), mean_tx=round(float(T.loc[d.index, 'txcellnet'].mean()), 4), mean_m=round(float(T.loc[d.index, k].mean()), 4),
                      wins=int((d > 0).sum()), diff=round(float(d.mean()), 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                      p=float(wilcoxon(d.values, alternative='greater').pvalue))
    return out


S = {}
for tier, TT in (('within', WITHIN), ('cross', CROSS)):
    for m in MET:
        S[f'{tier}_{m}'] = dict(mean=TT[m].mean().round(4).to_dict(), compare=compare(TT[m]))
        TT[m].to_csv(f'{OUT}/{tier}_{m}_per_dataset.csv')
nok = [c for c in WITHIN['spearman'].index if c not in K5]
S['within_spearman_no5K'] = dict(n=len(nok), mean=WITHIN['spearman'].loc[nok].mean().round(4).to_dict(), compare=compare(WITHIN['spearman'].loc[nok]))
# single TxCellNet networks (mean of the six per dataset, and each network) against single published models, cross tier
T1 = cross_table('spearman', dict(XMAP, txcellnet=SINGLE, nichetrans='nichetrans_off', mlp='mlp_pm'), TEST + FRESH)
S['cross_single'] = dict(mean=T1.mean().round(4).to_dict(), compare=compare(T1))
CS = CR['spearman'].reindex(TEST + FRESH)
S['cross_each_single_network'] = {s: {k: int((CS[s] > T1[k]).sum()) for k in ['nichetrans', 'dgat', 'spamosaic', 'totalvi', 'mlp']} for s in SINGLE if s in CS}
json.dump(S, open(f'{OUT}/robustness.json', 'w'), indent=1)
for k, v in S.items():
    if 'compare' in v:
        print(f'\n== {k}'); print(' '.join(f'{m}={x:.3f}' for m, x in v['mean'].items()))
        for m, c in v['compare'].items(): print(f'  vs {m:10s} n={c["n"]:2d} wins={c["wins"]:2d} diff={c["diff"]:+.3f} CI={c["ci"]} p={c["p"]:.1e}')
print('\n== each single network: wins against single models (of 24)', S['cross_each_single_network'])
