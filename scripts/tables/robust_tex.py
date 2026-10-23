"""LaTeX body of Supplementary Table S5 from robustness.json (robustness.py): mean over datasets of the per-dataset median
over markers (Spearman, Pearson, AUROC) per method and tier; in parentheses the one-sided Wilcoxon signed-rank p (paired by
dataset) for TxCellNet being better. Best value per column in bold.
usage: robust_tex.py ROBUSTNESS_JSON OUT_TEX
"""
import json, math, os, sys
import pandas as pd

S = json.load(open(sys.argv[1]))
# means from the full-precision per-dataset tables next to the JSON (the JSON keeps 4 decimals: rounding those again to 3
# decimals gave e.g. 0.6885 -> 0.689 for a mean of 0.68846)
D = os.path.dirname(os.path.abspath(sys.argv[1]))
NAMES = [('txcellnet', 'TxCellNet'), ('nichetrans', 'NicheTrans'), ('dgat', 'DGAT'), ('spamosaic', 'SpaMosaic'), ('totalvi', 'totalVI'),
         ('mlp', 'MLP'), ('ridge', 'Ridge'), ('knn', 'kNN mapping'), ('celltype', 'Cell-type lookup'), ('own_mRNA', 'Cognate mRNA')]
COLS = [(t, m) for t in ('within', 'cross') for m in ('spearman', 'pearson', 'auroc')]


def fp(p):
    if p >= 1e-3: return f'${p:.2g}$' if p < 0.1 else f'${p:.2f}$'
    e = math.floor(math.log10(p)); m = round(p / 10 ** e)
    if m == 10: m, e = 1, e + 1
    return f'${m}{{\\cdot}}10^{{{e}}}$'


MEAN = {c: pd.read_csv(f'{D}/{c[0]}_{c[1]}_per_dataset.csv', index_col=0).mean() for c in COLS}
best = {c: MEAN[c].max() for c in COLS}
L = [r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
     r' & \multicolumn{3}{c}{within datasets (34)} & \multicolumn{3}{c}{across datasets (24)}\\',
     r'\cmidrule(lr){2-4}\cmidrule(l){5-7}', r'Method & Spearman & Pearson & AUROC & Spearman & Pearson & AUROC\\', r'\midrule']
for k, name in NAMES:
    cells = []
    for c in COLS:
        d = S[f'{c[0]}_{c[1]}']; v = MEAN[c][k]; s = f'\\textbf{{{v:.3f}}}' if round(v, 3) == round(best[c], 3) else f'{v:.3f}'
        cmp_ = d['compare'].get(k); cells.append(f'{s} ({fp(cmp_["p"])})' if k != 'txcellnet' and cmp_ else s)
    L.append(f'{name} & ' + ' & '.join(cells) + r'\\')
L += [r'\bottomrule', r'\end{tabular}']
open(sys.argv[2], 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
