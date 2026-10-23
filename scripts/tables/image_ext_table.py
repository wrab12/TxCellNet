"""Supplementary table of image_metrics_ext.py: per kind of readout, mean over datasets of the per-dataset median over
markers of the pixel Pearson and the within-cell Pearson (pixel minus its cell's mean); painted per-cell predictions have
within-cell Pearson 0 by construction. Also prints the numbers quoted in the text.
usage: image_ext_table.py IMAGE_METRICS_EXT_CSV OUT_TEX
"""
import sys
import numpy as np, pandas as pd

D = pd.read_csv(sys.argv[1])
GR = [('Xenium segmentation stains', lambda c: c.startswith('xen')), ('IMC (CRLM)', lambda c: c.startswith('crlm')),
      ('COMET (liver, tonsil)', lambda c: c.startswith('duchini')), ('All', lambda c: True)]
PIX = [('TxCellNet (within)', 'TxC'), ('NicheTrans (within)', 'NT'), ('DGAT (within)', 'DGAT'), ('TxCellNet (cross)', 'TxC'),
       ('NicheTrans (cross)', 'NT'), ('cognate mRNA', 'mRNA')]
WIT = [('TxCellNet (within)', 'TxC'), ('TxCellNet (cross)', 'TxC'), ('cognate mRNA', 'mRNA')]
P = D.groupby(['case', 'method'])[['pixel_r', 'within_r']].median()
pix, wit = P.pixel_r.unstack('method'), P.within_r.unstack('method')
L = [r'\begin{tabular}{@{}lrrrrrrrrrr@{}}', r'\toprule',
     r' & & \multicolumn{6}{c}{pixel Pearson} & \multicolumn{3}{c}{within-cell Pearson}\\',
     r'\cmidrule(lr){3-8}\cmidrule(l){9-11}',
     r' & & \multicolumn{3}{c}{trained within} & \multicolumn{2}{c}{on kidney} & & within & kidney & \\',
     r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}',
     r'Readout & $n$ & ' + ' & '.join(s for _, s in PIX) + ' & ' + ' & '.join(s for _, s in WIT) + r'\\', r'\midrule']
for name, f in GR:
    cs = [c for c in pix.index if f(c)]
    vp = [pix.loc[cs, m].mean() for m, _ in PIX]; vw = [wit.loc[cs, m].mean() for m, _ in WIT]
    bw, bx = max(vp[:3]), max(vp[3:5])
    cells = [(f'\\textbf{{{v:.2f}}}' if b is not None and round(v, 2) == round(b, 2) else f'{v:.2f}') for v, b in zip(vp, [bw] * 3 + [bx] * 2 + [None])]
    cells += [f'{v:.2f}' for v in vw]
    if name == 'All': L.append(r'\midrule')
    L.append(f'{name} & {len(cs)} & ' + ' & '.join(cells) + r'\\')
L += [r'\bottomrule', r'\end{tabular}']
open(sys.argv[2], 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
both = wit[['TxCellNet (cross)', 'cognate mRNA']].dropna()
print('within-cell: TxCellNet (kidney) > mRNA density on', int((both.iloc[:, 0] > both.iloc[:, 1]).sum()), 'of', len(both))
both = wit[['TxCellNet (within)', 'cognate mRNA']].dropna()
print('within-cell: TxCellNet (within) > mRNA density on', int((both.iloc[:, 0] > both.iloc[:, 1]).sum()), 'of', len(both))
for a, b in (('TxCellNet (within)', 'NicheTrans (within)'), ('TxCellNet (within)', 'DGAT (within)'), ('TxCellNet (cross)', 'NicheTrans (cross)')):
    x = pix[[a, b]].dropna(); print(f'pixel: {a} > {b} on', int((x[a] > x[b]).sum()), 'of', len(x))
