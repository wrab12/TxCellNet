"""Supplementary Fig. S2: which proteins transfer. Input: protein_table.csv of scripts/figdata/protein_table.py
(NicheTrans: nichetrans_off_e3).
(a) TxCellNet's cell-level Spearman per protein: kidney held-out stripes | kidney cross-patient (trained on one section, tested
    on the other) | CRLM, IMC (test ROIs, median) | liver / tonsil, COMET (test cores, median); the cognate mRNA on CRLM; the
    difference TxCellNet minus NicheTrans on CRLM
(b) mean per protein category per setting
(c) per protein on CRLM: TxCellNet (filled) and NicheTrans (open) against the cognate mRNA
Every printed glyph >= 10 pt (figstyle.print_min10); panel letters 11 pt. To fit 10 pt the panels are re-arranged: a on top
(colour keys in the gutter to its right), b and c side by side below.
usage: figS2_proteins.py TABLE_CSV OUT_PREFIX
"""
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import figstyle
figstyle.print_min10()
from figstyle import W, LS, label, axes_in, save, INK, HERO, COL
from benchtable import SHADE

TAB, OUT = sys.argv[1:3]
FS = 10          # every text item
H = pd.read_csv(TAB, index_col=0)
H.columns = [c.replace('\n', ' ') for c in H.columns]
CAT = {'Immune\nlineage': ['CD20', 'CD3E', 'CD4', 'CD8A', 'CD45', 'CD45RO', 'CD45RA', 'CD68', 'CD163', 'CD11c', 'CD16'],
       'Epithelial,\ntumor': ['PanCK', 'E-Cadherin', 'Beta-catenin'],
       'Stromal,\nvascular': ['Vimentin', 'alphaSMA', 'CD31'],
       'Cell state,\ncheckpoint': ['Ki-67', 'PD-1', 'PD-L1', 'VISTA', 'GranzymeB']}
CATCOL = ['#7B5EA7', '#3E9C8F', '#C49A2C', '#B0567A']
ORDER = [p for v in CAT.values() for p in v]; H = H.reindex(ORDER)
PRETTY = {'E-Cadherin': 'E-cadherin', 'Beta-catenin': 'β-catenin', 'alphaSMA': 'αSMA', 'GranzymeB': 'Granzyme B', 'CD8A': 'CD8',
          'CD3E': 'CD3'}
SET = ['kidney held-out', 'kidney cross-patient', 'CRLM (IMC)', 'tonsil / liver (COMET)']
HEADS = ['Kidney\nheld-out', 'Kidney\nother pt.', 'CRLM\n(IMC)', 'Liver,\ntonsil\n(COMET)', 'CRLM\ncognate\nmRNA',
         'CRLM\nTxCellNet −\nNicheTrans']
COG, DIF = 'CRLM cognate mRNA', 'CRLM minus NicheTrans'
DIV = mcolors.LinearSegmentedColormap.from_list('div', [COL['nichetrans'], '#F4F4F4', HERO])

# ---- layout (inches from the top-left) ----
Hh = 7.25
CW, RH, GAP = 0.62, 0.17, 0.25           # cell width (in), row height (in), gap before the mRNA and difference columns (cells)
XA, YA = 1.0, 0.80                       # a: heatmap left edge, top of the first row
YB, PH = 5.02, 1.75                      # b, c: plot top and height
fig = plt.figure(figsize=(W, Hh))

# ---- a: per protein ----
label(fig, 0.0, 0.03, 'a'); fig.text(0.17 / W, 1 - 0.03 / Hh, 'TxCellNet per protein (cell-level Spearman)', fontsize=FS, va='top', color=INK)
xs = [0, 1, 2, 3, 4 + GAP, 5 + 2 * GAP]
nrow = len(ORDER)
ax = axA = axes_in(fig, XA, YA, (xs[-1] + 1) * CW, nrow * RH)
ax.set_xlim(0, xs[-1] + 1); ax.set_ylim(nrow - 0.5, -0.5); ax.axis('off')
for i, p in enumerate(ORDER):
    for j, c in enumerate(SET + [COG, DIF]):
        v = H.loc[p, c]
        if not np.isfinite(v):
            ax.add_patch(Rectangle((xs[j] + 0.03, i - 0.45), 0.94, 0.9, fc='#FAFAFA', ec='none')); continue
        colr = DIV((np.clip(v, -0.3, 0.3) + 0.3) / 0.6) if c == DIF else SHADE(np.clip(v, 0, 0.9) / 0.9)
        lum = np.dot(mcolors.to_rgb(colr), [0.299, 0.587, 0.114])
        ax.add_patch(Rectangle((xs[j] + 0.03, i - 0.45), 0.94, 0.9, fc=colr, ec='none'))
        ax.text(xs[j] + 0.5, i, (f'{v:+.2f}' if c == DIF else f'{v:.2f}').replace('-', '−'), ha='center', va='center_baseline', fontsize=FS,
                color='white' if lum < 0.5 else INK)
    ax.annotate(PRETTY.get(p, p), xy=(0, i), xycoords='data', xytext=(-4, 0), textcoords='offset points', ha='right',
                va='center_baseline', fontsize=FS, color=INK, annotation_clip=False)
k = 0
x_strip = (0.06 - XA) / CW                # category strip 0.06 in from the page edge
for (cat, ps), col in zip(CAT.items(), CATCOL):
    ax.add_patch(Rectangle((x_strip, k - 0.45), 0.08 / CW, len(ps) - 0.1, fc=col, ec='none', clip_on=False))
    k += len(ps)
for j, h in enumerate(HEADS):
    ax.annotate(h, xy=(xs[j] + 0.5, -0.5), xycoords='data', xytext=(0, 3), textcoords='offset points', ha='center',
                va='bottom', fontsize=FS, color=INK, linespacing=LS, annotation_clip=False)
# colour keys in the gutter right of the heatmap
XK, WK = XA + (xs[-1] + 1) * CW + 0.33, 0.9
for yk, cmap, lo, hi, ttl in [(YA + 0.22, SHADE, 0, 0.9, 'Spearman'), (YA + 0.98, DIV, -0.3, 0.3, 'Difference')]:
    cax = axes_in(fig, XK, yk, WK, 0.08)
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect='auto', extent=[lo, hi, 0, 1])
    cax.set_yticks([]); cax.set_xticks([lo, (lo + hi) / 2, hi]); cax.tick_params(axis='x', labelsize=FS, length=2, pad=1.5)
    for s in cax.spines.values(): s.set_visible(False)
    cax.set_title(ttl, fontsize=FS, pad=3, color=INK)

# ---- b: category means ----
YL = YB - 0.3
label(fig, 0.0, YL, 'b'); fig.text(0.17 / W, 1 - YL / Hh, 'By protein category', fontsize=FS, va='top', color=INK)
ax = axB = axes_in(fig, 0.6, YB, 1.95, PH)
for (cat, ps), col in zip(CAT.items(), CATCOL):
    ax.plot(range(4), [H.loc[ps, s].mean() for s in SET], marker='o', ms=4.2, lw=1.4, color=col, label=cat, mec='white', mew=0.4)
ax.set_xticks(range(4), ['Kidney', 'Kidney\nother pt.', 'CRLM\n(IMC)', 'Liver,\ntonsil'], fontsize=FS, linespacing=LS)
ax.set_xlim(-0.3, 3.3)
ax.set_ylabel('Mean Spearman'); ax.set_ylim(-0.1, 0.95); ax.axhline(0, color='#BDBDBD', lw=0.6)
ax.legend(fontsize=FS, loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=1.3, labelspacing=0.7, borderaxespad=0)

# ---- c: beyond the cognate transcript on CRLM ----
XC = 4.25
label(fig, XC - 0.62, YL, 'c'); fig.text((XC - 0.45) / W, 1 - YL / Hh, 'Beyond the cognate transcript (CRLM)', fontsize=FS,
                                         va='top', color=INK)
ax = axC = axes_in(fig, XC, YB, 2.15, PH)
G = H[[COG, 'CRLM (IMC)', DIF]].dropna()
xo, yt = G[COG], G['CRLM (IMC)']; yn = yt - G[DIF]
ax.plot([-0.2, 1.0], [-0.2, 1.0], color='#BDBDBD', lw=0.6, zorder=1)
ax.vlines(xo, np.minimum(yt, yn), np.maximum(yt, yn), color='#D5D5D5', lw=0.7, zorder=2)
ax.scatter(xo, yn, s=16, facecolors='white', edgecolors=COL['nichetrans'], linewidths=0.9, zorder=3, label='NicheTrans')
ax.scatter(xo, yt, s=19, c=HERO, edgecolors='white', linewidths=0.4, zorder=4, label='TxCellNet')
# labels placed in empty space at 10 pt: (anchor: TxCellNet 't' or NicheTrans 'n' point, offset pt, ha, va)
# CD68 sits just below-left of the CD45 pair, so its label goes up-left into empty space with a thin leader to its own point
OFF = {'VISTA': ('t', (-4, -2.5), 'right', 'center'), 'CD68': ('t', (-13, 7), 'right', 'bottom'),
       'PanCK': ('t', (0, 4), 'center', 'bottom'), 'alphaSMA': ('n', (5, 0), 'left', 'center'),
       'Vimentin': ('t', (4, 0), 'left', 'center')}
LEADER = {'CD68': dict(arrowstyle='-', lw=0.5, color='#9A9A9A', shrinkA=0, shrinkB=2.6, relpos=(1, 0.15))}
for p in ['VISTA', 'CD68', 'CD45RO', 'alphaSMA', 'PanCK', 'Vimentin']:
    if p in G.index:
        pt, off, ha, va = OFF.get(p, ('t', (3, -1), 'left', 'top'))
        ax.annotate(PRETTY.get(p, p), (xo[p], yt[p] if pt == 't' else yn[p]), xytext=off, textcoords='offset points',
                    fontsize=FS, color=INK, ha=ha, va=va, arrowprops=LEADER.get(p))
ax.set_xlim(-0.3, 1.0); ax.set_ylim(-0.2, 1.0); ax.patch.set_visible(False)  # Vimentin label ends just past x = 1
ax.set_xticks([0, 0.4, 0.8]); ax.set_yticks([0, 0.4, 0.8])
ax.set_xlabel('Cognate mRNA (Spearman)'); ax.set_ylabel('Model (Spearman)')
ax.legend(fontsize=FS, loc='upper left', handletextpad=0.2, borderaxespad=0.1)
save(fig, OUT, {'a_per_protein': H.round(4)}, panels={'a': axA, 'b': axB, 'c': axC}, row_groups=[['b', 'c']])
print(H.round(2).to_string())
