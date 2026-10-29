"""Supplementary Fig. S6: paired comparison per section in the three settings. For every
section (or pair of sections) the median cell-level Spearman over markers of TxCellNet minus that of each published method
(dots: sections or pairs; line: median; number above: sections or pairs where TxCellNet is better; one-sided Wilcoxon
signed-rank tests, alternative TxCellNet better, in the source data).
(a) same section, 34 sections (Supplementary Table S4; TxCellNet one clean network per section);
(b) another section of the same dataset, 50 pairs (Supplementary Table S7; TxCellNet = clean + randomized network of the
    training section; NicheTrans seed 1); SpaMosaic diverged on one pair (49);
(c) kidney-trained, 24 test and fresh sections (Supplementary Table S3; six-network TxCellNet ensemble, NicheTrans three-seed
    ensemble).
Every printed glyph >= 10 pt (figstyle.print_min10); panel letters 11 pt.
usage: figS6_paired.py DATA_DIR OUT_PREFIX
"""
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import figstyle
figstyle.print_min10()
from figstyle import W, label, axes_in, save, INK, COL, MK, PUBLISHED

DD, OUT = sys.argv[1:3]
FS = 10
KEY = {'txcellnet': 'txcellnet_cr', 'nichetrans': 'nichetrans_off'}
num = lambda v: format(v, 'g').replace('-', '−')

PW = pd.read_csv(f'{DD}/tables/fig_within_per_dataset.csv', index_col=0)          # 34 sections
PC = pd.read_csv(f'{DD}/tables/fig_cross_per_dataset.csv', index_col=0)           # 24 sections
XD = pd.read_csv(f'{DD}/final/summary/xsec_datasets.csv')
XS = XD[XD.tier == 'cross-section']
PX = pd.DataFrame({m: XS[XS.method == KEY.get(m, m)].groupby(['train', 'test']).spearman.mean()
                   for m in ['txcellnet'] + PUBLISHED}).dropna(subset=['txcellnet'])   # 50 pairs
assert len(PW) == 34 and len(PX) == 50 and len(PC) == 24
SETTINGS = [('a', 'Same section (34 sections)', 'sections', PW), ('b', 'Other section (50 pairs)', 'pairs', PX),
            ('c', 'Kidney-trained (24 sections)', 'sections', PC)]
YMIN, YMAX = -0.44, 0.92                                                           # every unit is shown (asserted)
assert all(YMIN < (T['txcellnet'] - T[m]).min() and (T['txcellnet'] - T[m]).max() < YMAX for *_, T in SETTINGS for m in PUBLISHED)

H = 2.42
Y0, YN, YP, PH = 0.03, 0.40, 0.56, 1.50          # letter top, win-count row, plot top, plot height (in)
XP, PWID, GUT = 0.62, 1.86, 0.12                 # first plot left, plot width, gutter (plots share the y axis)
fig = plt.figure(figsize=(W, H))
axes, tabs, diffs = {}, {}, []
for j, (pid, title, unit, T) in enumerate(SETTINGS):
    x0 = XP + j * (PWID + GUT)
    lx = 0.0 if j == 0 else x0 - 0.04
    label(fig, lx, Y0, pid); fig.text((lx + 0.17) / W, 1 - (Y0 + 0.01) / H, title, fontsize=FS, va='top', color=INK)
    ax = axes[pid] = axes_in(fig, x0, YP, PWID, PH)
    rows = []
    for k, m in enumerate(PUBLISHED):
        d = (T['txcellnet'] - T[m]).dropna()
        diffs += [dict(setting=title, unit=' -> '.join(i) if isinstance(i, tuple) else i, method=m, difference=v) for i, v in d.items()]
        dv = d.sort_values().values
        off = (np.arange(len(dv)) % 5 - 2) * 0.08                              # deterministic spread within the strip
        ax.scatter(k + off, dv, s=8, marker=MK[m], c=COL[m], edgecolors='none', zorder=3)
        ax.plot([k - 0.3, k + 0.3], [np.median(dv)] * 2, color=INK, lw=1.1, zorder=4, solid_capstyle='butt')
        w, n = int((dv > 0).sum()), len(dv)
        rows.append({'method': m, 'wins': w, unit: n, 'p_one_sided': wilcoxon(dv, alternative='greater').pvalue})
        fig.text((x0 + (k + 0.5) / 4 * PWID) / W, 1 - YN / H, f'{w}/{n}', fontsize=FS, ha='center', va='center', color=INK)
    tabs[pid] = pd.DataFrame(rows).set_index('method')
    ax.axhline(0, color='#9A9A9A', lw=0.6, zorder=1)
    ax.set_xlim(-0.5, 3.5); ax.set_xticks(range(4), ['NT', 'DGAT', 'SpaM.', 'tVI'])
    ax.set_ylim(YMIN, YMAX); ax.set_yticks([-0.4, 0, 0.4, 0.8], [num(v) for v in (-0.4, 0, 0.4, 0.8)])
    ax.grid(axis='y', color='#EEEEEE', lw=0.5); ax.set_axisbelow(True)
    if j: ax.tick_params(axis='y', labelleft=False)
    else: ax.set_ylabel('TxCellNet minus method')
save(fig, OUT, {'a_same_section': tabs['a'], 'b_other_section': tabs['b'], 'c_kidney_only': tabs['c'],
                'differences': pd.DataFrame(diffs).set_index(['setting', 'unit', 'method']).round(6)},
     panels=axes, row_groups=[['a', 'b', 'c']])
pd.set_option('display.width', 200)
for pid, t in tabs.items(): print(pid); print(t.to_string())
