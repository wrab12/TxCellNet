"""Summary of the spot-level evaluation (grid_eval.py): per-dataset and per-group tables, wins and one-sided Wilcoxon tests
of TxCellNet against each method, LaTeX tables for the supplement.
usage: grid_summary.py [OUT]   (default results/bench/grid) -> OUT/summary/
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

from txcellnet.bench import GRID_GROUPS_WITHIN as GROUPS_WITHIN, GRID_GROUPS_CROSS as GROUPS_CROSS, GRID_METHODS as METHODS

OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/bench/grid'
os.makedirs(f'{OUT}/summary', exist_ok=True)
HEAD = {'Kidney (IF)': ('Kidney', '(IF)'), 'CRLM (IMC)': ('CRLM', '(IMC)'), 'Liver, tonsil (COMET)': ('Liver, tonsil', '(COMET)'),
        'Breast (stains)': ('Breast', '(stains)'), 'Lung, pancreas (stains)': ('Lung, panc.', '(stains)'),
        'Lymph node, fresh (stains)': ('LN, fresh', '(stains)'), 'Lymph node (stains)': ('Lymph node', '(stains)'),
        'Fresh (stains)': ('Fresh', '(stains)')}

D = pd.concat([pd.read_csv(f).assign(setting='within') for f in glob.glob(f'{OUT}/within/*.csv')]
              + [pd.read_csv(f).assign(setting='within' if os.path.basename(f).startswith('kidney') else 'cross')
                 for f in glob.glob(f'{OUT}/cross/*.csv')])
stats, tabs = {}, {}
for setting, groups in (('within', GROUPS_WITHIN), ('cross', GROUPS_CROSS)):
    cases = [c for _, cs in groups for c in cs]
    S = D[(D.setting == setting) & D.case.isin(cases)]
    T = S.groupby(['level', 'case', 'method']).spearman.median().unstack('method')
    info = S.groupby('case')[['spots_test', 'cells_per_spot']].first().reindex(cases)
    stats[f'{setting}_missing'] = [c for c in cases if c not in set(S.case)]
    stats[f'{setting}_spots'] = dict(test_spots_median=float(info.spots_test.median()), test_spots_min=float(info.spots_test.min()),
                                     test_spots_max=float(info.spots_test.max()), cells_per_spot_median=float(info.cells_per_spot.median()),
                                     cells_per_spot_range=[float(info.cells_per_spot.min()), float(info.cells_per_spot.max())])
    for level in ('cell', 'spot'):
        Tl = T.loc[level].reindex(cases)[[m for m, _ in METHODS if m in T.loc[level].columns]]
        Tl.join(info).to_csv(f'{OUT}/summary/{setting}_{level}_per_dataset.csv')
        G = pd.DataFrame({g: Tl.reindex(cs).mean() for g, cs in groups}); G['All'] = Tl.mean()
        G.to_csv(f'{OUT}/summary/{setting}_{level}_groups.csv'); tabs[(setting, level)] = G
        st = {'n': int(Tl.txcellnet.notna().sum()), 'best_count': int((Tl.txcellnet >= Tl.max(1) - 1e-12).sum())}
        for m in Tl.columns:
            if m == 'txcellnet': continue
            d = (Tl.txcellnet - Tl[m]).dropna()
            st[m] = dict(wins=f'{int((d > 0).sum())}/{len(d)}', mean_diff=round(float(d.mean()), 4),
                         p_one_sided=float(wilcoxon(d, alternative='greater').pvalue) if len(d) >= 5 else None)
        if 'dgat_spot' in Tl and 'dgat_cell' in Tl:
            d = (Tl.dgat_spot - Tl.dgat_cell).dropna()
            st['dgat_spot_minus_dgat_cell'] = dict(better=f'{int((d > 0).sum())}/{len(d)}', mean_diff=round(float(d.mean()), 4),
                                                   p_two_sided=float(wilcoxon(d).pvalue) if len(d) >= 5 else None)
        stats[f'{setting}_{level}'] = st

json.dump(stats, open(f'{OUT}/summary/stats.json', 'w'), indent=1)
for k, G in tabs.items(): print(k, '\n', G.round(3).to_string(), '\n')
print(json.dumps(stats, indent=1))

for setting, groups in (('within', GROUPS_WITHIN), ('cross', GROUPS_CROSS)):
    G = tabs[(setting, 'spot')]; best = G.max(0); cols = list(G.columns)
    h1 = ' & '.join(HEAD.get(c, (c, ''))[0] for c in cols); h2 = ' & '.join(HEAD.get(c, (c, ''))[1] for c in cols)
    n = ' & '.join(str(len(cs)) for _, cs in groups) + f' & {sum(len(cs) for _, cs in groups)}'
    L = [f'\\begin{{tabular}}{{@{{}}l{"r" * len(cols)}@{{}}}}', '\\toprule', f'Method & {h1}\\\\', f' & {h2}\\\\', f'sections & {n}\\\\', '\\midrule']
    for m, name in METHODS:
        if m not in G.index: continue
        L.append(f'{name} & ' + ' & '.join((f'\\textbf{{{v:.3f}}}' if abs(v - best[c]) < 5e-4 else f'{v:.3f}') if np.isfinite(v) else '--'
                                           for c, v in G.loc[m].items()) + '\\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    open(f'{OUT}/summary/table_grid_{setting}.tex', 'w').write('\n'.join(L) + '\n')
print('tables written')
