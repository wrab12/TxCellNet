"""Supplementary Table S1 (semi-synthetic benchmark) from the evaluation rows: results/sim/ens/sim_rows_*.csv (TxCellNet
networks and the ridge baselines of sim_eval.py) and results/sim/sota/sim_rows_*.csv (the published methods of
sim_sota.py; NicheTrans trained as in its official tutorial, which replaces the shortened NicheTrans of sim_eval.py).
Mean over six channels and two sections. Also prints the numbers quoted in the main text.
usage: sim_table.py OUT_TEX
"""
import glob, sys
import numpy as np, pandas as pd

E = pd.concat([pd.read_csv(f) for f in glob.glob('results/sim/ens/sim_rows_*.csv')])
E = E[E.method != 'NicheTrans']
S = pd.concat([pd.read_csv(f) for f in glob.glob('results/sim/sota/sim_rows_*.csv')])
D = pd.concat([E, S])
ROWS = [('TxCellNet:ens_cr', 'TxCellNet (clean + randomized)'), ('TxCellNet:sim_plain', 'TxCellNet, clean network'),
        ('TxCellNet:sim_aug', 'TxCellNet, randomized network'), ('TxCellNet:sim_mix', 'TxCellNet, mixed network'),
        ('NicheTrans', 'NicheTrans'), ('DGAT', 'DGAT'), ('SpaMosaic', 'SpaMosaic'), ('totalVI', 'totalVI'),
        ('ridge_niche', 'Ridge + neighborhood'), ('ridge', 'Ridge')]
ROWS = [(k, n) for k, n in ROWS if k in set(D.method)]
NC = ['niche_r15', 'niche_r40', 'niche_r80']
M = D.groupby(['perturb', 'method'])[['cell_spearman', 'pixel_pearson', 'within_cell']].mean()
N = D[(D.perturb == 'none') & D.channel.isin(NC)].groupby(['method', 'channel']).cell_spearman.mean().unstack('channel')[NC]


def g(p, k, col='cell_spearman'):
    return M.loc[(p, k), col] if (p, k) in M.index else np.nan


f2 = lambda v: '--' if not np.isfinite(v) else ('0' if abs(v) < 5e-3 else f'{v:.2f}')
L = [r'\begin{tabular}{@{}lcccccccccc@{}}', r'\toprule',
     r' & \multicolumn{3}{c}{Unperturbed} & \multicolumn{2}{c}{\begin{tabular}[b]{@{}c@{}}Molecules\\kept\end{tabular}} & Masks & Cells'
     r' & \multicolumn{3}{c}{Niche channels}\\',
     r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(l){9-11}',
     r'Method & Cell & Pixel & Within-cell & 50\% & 25\% & dilated & merged & 15\,$\mu$m & 40\,$\mu$m & 80\,$\mu$m\\', r'\midrule']
for k, name in ROWS:
    v = [g('none', k), g('none', k, 'pixel_pearson'), g('none', k, 'within_cell'), g('thin50', k), g('thin25', k), g('dilate2', k), g('merge20', k)]
    v += list(N.loc[k].values) if k in N.index else [np.nan] * 3
    L.append(f'{name} & ' + ' & '.join(f2(x) for x in v) + r'\\')
L += [r'\bottomrule', r'\end{tabular}']
open(sys.argv[1], 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
per = [k for k, _ in ROWS if not k.startswith('TxCellNet')]
best = max(per, key=lambda k: g('none', k))
print('best per-cell method (unperturbed cell Spearman):', best, round(g('none', best), 3), 'pixel', round(g('none', best, 'pixel_pearson'), 3))
for k in per: print(f'{k:12s} none {g("none", k):.3f} pix {g("none", k, "pixel_pearson"):.3f} thin25 {g("thin25", k):.3f} niche {N.loc[k].min():.3f}-{N.loc[k].max():.3f}')
