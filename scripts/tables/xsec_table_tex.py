"""LaTeX table of the cross-section tier for the supplement: rows = methods, columns = dataset groups; entry = mean over
the (training section, test section) pairs of the group of the median cell-level Spearman over markers on the test
section's held-out stripes. Best per column in bold. Kidney: single-section models.
usage: xsec_table_tex.py GROUPS_CSV OUT_TEX
"""
import sys
import numpy as np, pandas as pd

G = pd.read_csv(sys.argv[1]); X = G[G.tier == 'cross-section']
COLS = [('breast, same patient', 'Br.\\ same pt.'), ('breast, other patient', 'Br.\\ other pt.'), ('CRLM tumour 3, other ROI', 'CRLM T3 ROI'),
        ('CRLM, other section', 'CRLM other'), ('COMET, other liver core', 'COMET liver'), ('COMET, liver <-> tonsil', 'COMET liv.--ton.'),
        ('kidney, other patient', 'Kidney')]
ROWS = [('txcellnet_cr', 'TxCellNet (clean + randomized)'), ('txcellnet', '\\quad clean network only'), ('txcellnet_rand', '\\quad randomized network only'),
        ('nichetrans_off', 'NicheTrans'), ('dgat', 'DGAT'), ('spamosaic', 'SpaMosaic'), ('totalvi', 'totalVI'), ('mlp', 'MLP'), ('ridge', 'Ridge'),
        ('knn', 'Reference mapping'), ('celltype', 'Cell-type lookup'), ('own_mRNA', 'Cognate mRNA')]
V = {(g, m): X[(X.group == g) & (X.method == m)].spearman for g, _ in COLS for m, _ in ROWS}
V = {k: (v.mean() if len(v) else np.nan) for k, v in V.items()}
best = {g: np.nanmax([V[(g, m)] for m, _ in ROWS if m not in ('txcellnet', 'txcellnet_rand')]) for g, _ in COLS}
npairs = {g: int(X[X.group == g].n.max()) if len(X[X.group == g]) else 0 for g, _ in COLS}
lines = ['\\begin{tabular}{@{}l' + 'r' * len(COLS) + '@{}}', '\\toprule',
         'Method & ' + ' & '.join(h for _, h in COLS) + '\\\\',
         'pairs & ' + ' & '.join(str(npairs[g]) for g, _ in COLS) + '\\\\', '\\midrule']
for m, name in ROWS:
    cells = []
    for g, _ in COLS:
        v = V[(g, m)]
        s = '--' if not np.isfinite(v) else f'{v:.3f}'
        if np.isfinite(v) and m not in ('txcellnet', 'txcellnet_rand') and abs(v - best[g]) < 5e-4: s = f'\\textbf{{{s}}}'
        cells.append(s)
    lines.append(name + ' & ' + ' & '.join(cells) + '\\\\')
    if m == 'txcellnet_rand': lines.append('\\midrule')
lines += ['\\bottomrule', '\\end{tabular}']
open(sys.argv[2], 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
