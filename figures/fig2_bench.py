"""Figure 2: cross-dataset generalization. Every method is trained on the two kidney sections only
and applied to the 24 test and fresh sections of the other datasets. (a) Cleveland dot plot of the mean cell-level Spearman per
group of sections for TxCellNet, the four published methods and MLP (the strongest baseline in this setting), last row: all 24
sections; (b) per section, TxCellNet minus each published method (dots: sections; line: median; number: sections where
TxCellNet is better; one-sided Wilcoxon signed-rank tests); (c) the three settings on identical test cells (held-out stripes of
the 22 breast, CRLM and COMET sections): trained on the same section, on another section of the same dataset, or on kidney
(slopegraph). The same-section and other-section group results are in Supplementary Fig. S5 (figS5_tables.py).
Every printed glyph >= 10 pt (figstyle.print_min10); panel letters 11 pt. To fit one row at 10 pt the group names of a sit
above their rows, and the star of c is labelled "6-net ensemble" (the six-network TxCellNet of a, as in the caption).
usage: fig2_bench.py DATA_DIR OUT_PREFIX
"""
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon
import figstyle
figstyle.print_min10()
from figstyle import W, LS, label, axes_in, save, INK, COL, LAB, MK, HERO, HERO_LIGHT, PUBLISHED

DD, OUT = sys.argv[1:3]
SHOWN = ['txcellnet', 'nichetrans', 'dgat', 'totalvi', 'spamosaic', 'mlp']
KEY = {'txcellnet': 'txcellnet_cr', 'nichetrans': 'nichetrans_off'}
FS = 10          # every text item (rcParams already 10 after print_min10)

PC = pd.read_csv(f'{DD}/tables/fig_cross_per_dataset.csv', index_col=0)           # 24 sections, kidney-trained
assert len(PC) == 24
GC = [('Breast (9)', [c for c in PC.index if c.startswith('xen_breast_S')]), ('Lymph node (1)', ['xen_lymphnode_5k']),
      ('CRLM (4)', [c for c in PC.index if c.startswith('crlm')]), ('Liver, tonsil (3)', [c for c in PC.index if c.startswith('duchini')]),
      ('Fresh (7)', ['xen_skin_5k', 'xen_prostate_5k', 'xen_ovary_5k', 'xen_ovary_v1ff', 'xen_breast_5k', 'xen_cervix_5k', 'xen_gbm_v1'])]
assert sum(len(c) for _, c in GC) == 24
TC = pd.DataFrame({h: PC.loc[cs].mean() for h, cs in GC}); TC['All (24)'] = PC.mean()
BASE = ['mlp', 'ridge', 'knn', 'celltype', 'own_mRNA']
assert TC.loc[BASE, 'All (24)'].astype(float).idxmax() == 'mlp'                    # MLP is the strongest baseline here

H = 3.0
MS_HERO = 5.2                          # TxCellNet diamond (key and a)
fig = plt.figure(figsize=(W, H))
rend = fig.canvas.get_renderer()

# ---- method key (top) ----
kx = 0.02                              # start 0.02 in from the page edge so the key diamond is not clipped
for m in SHOWN:
    fig.add_artist(Line2D([(kx + 0.05) / W], [1 - 0.1 / H], marker=MK[m], ms=MS_HERO if m == 'txcellnet' else 5.0, mfc=COL[m],
                          mec='white', mew=0.4, ls='none'))
    t = fig.text((kx + 0.14) / W, 1 - 0.1 / H, LAB[m], fontsize=FS, va='center', color=INK,
                 fontweight='bold' if m == 'txcellnet' else 'normal')
    kx += 0.14 + t.get_window_extent(rend).width / fig.dpi + 0.24

YT, YP, PH = 0.30, 0.66, 1.92          # panel-letter top, plot top, plot height (inches from the top)
XA, WA = 0.04, 1.10                    # a: plot (group names above the rows, so no label gutter)
XB, WB = 1.64, 1.71                    # b
XC, WC = 3.90, 1.16                    # c (direct labels to the right of the plot)


def title(x, s):
    fig.text(x / W, 1 - (YT + 0.01) / H, s, fontsize=FS, va='top', color=INK)


# ---- a: group means, kidney-trained ----
label(fig, 0.0, YT, 'a'); title(0.17, 'Kidney-trained')
axA = axes_in(fig, XA, YP, WA, PH)
rows = list(TC.columns); ys = np.arange(len(rows), dtype=float); ys[-1] += 0.5
for y, r in zip(ys, rows):
    v = TC.loc[SHOWN, r].astype(float)
    axA.plot([v.min(), v.max()], [y, y], color='#D0D0D0', lw=1.0, zorder=1, solid_capstyle='round')
    for m in SHOWN[::-1]:
        axA.plot(TC.loc[m, r], y, marker=MK[m], ms=MS_HERO if m == 'txcellnet' else 4.6, mfc=COL[m], mec='white', mew=0.4, ls='none',
                 zorder=6 if m == 'txcellnet' else 3)
    axA.annotate(r, xy=(0, y), xycoords=('axes fraction', 'data'), xytext=(0, 5.6), textcoords='offset points', ha='left',
                 va='bottom', fontsize=FS, color=INK, fontweight='bold' if r.startswith('All') else 'normal', annotation_clip=False)
axA.axhline(ys[-2] + 0.42, color='#BDBDBD', lw=0.6)
axA.set_ylim(ys[-1] + 0.35, -0.65); axA.set_yticks([])
axA.patch.set_visible(False); axA.spines['left'].set_visible(False); axA.set_xlim(0.12, 0.66); axA.set_xticks([0.2, 0.4, 0.6])
axA.set_xlabel('Spearman (mean)')

# ---- b: paired differences per section, kidney-trained ----
label(fig, 1.18, YT, 'b'); title(1.35, 'Paired by section')
axB = axes_in(fig, XB, YP, WB, PH)
YMIN, YMAX = -0.12, 0.5
pvals, wins = {}, {}
for k, m in enumerate(PUBLISHED):
    d = (PC['txcellnet'] - PC[m]).dropna().sort_values().values
    assert YMIN < d.min() and d.max() < YMAX
    off = (np.arange(len(d)) % 5 - 2) * 0.08
    axB.scatter(k + off, d, s=8, marker=MK[m], c=COL[m], edgecolors='none', zorder=3)
    axB.plot([k - 0.3, k + 0.3], [np.median(d)] * 2, color=INK, lw=1.1, zorder=4, solid_capstyle='butt')
    wins[m] = (int((d > 0).sum()), len(d)); pvals[m] = wilcoxon(d, alternative='greater').pvalue
    axB.text(k, YMAX + 0.015, f'{wins[m][0]}/{len(d)}', ha='center', va='bottom', fontsize=FS, color=INK)
axB.axhline(0, color='#9A9A9A', lw=0.6, zorder=1)
axB.set_xlim(-0.36, 3.36); axB.set_xticks(range(4), ['NT', 'DGAT', 'SpaM.', 'tVI'])
axB.set_ylim(YMIN, YMAX); axB.set_yticks([0, 0.2, 0.4], ['0', '0.2', '0.4'])
axB.set_ylabel('TxCellNet minus method'); axB.grid(axis='y', color='#EEEEEE', lw=0.5); axB.set_axisbelow(True)

# ---- c: three settings on identical test cells ----
XD = pd.read_csv(f'{DD}/final/summary/xsec_datasets.csv')
X = XD[~XD.test.str.startswith('kidney')]
TIERS = [('within', 'Same\nsection'), ('cross-section', 'Other\nsection'), ('cross-dataset', 'Kidney\nonly')]
LINES = ['txcellnet', 'nichetrans', 'dgat', 'totalvi', 'spamosaic', 'mlp', 'own_mRNA']


def tier_mean(key, tier):
    s = X[(X.tier == tier) & (X.method == key)]
    return s.groupby('test').spearman.mean().mean(), s.test.nunique()


TE = pd.DataFrame({t: {m: tier_mean(KEY.get(m, m), t)[0] for m in LINES} for t, _ in TIERS})
TE.loc['txcellnet_clean'] = [tier_mean('txcellnet', t)[0] for t, _ in TIERS]
TE.loc['txcellnet_ens6'] = [np.nan, np.nan, tier_mean('txcellnet_ens6', 'cross-dataset')[0]]
assert all(tier_mean('txcellnet_cr', t)[1] == 22 for t, _ in TIERS)
label(fig, 3.46, YT, 'c'); title(3.63, 'Three settings, same test cells')
ax = axC = axes_in(fig, XC, YP, WC, PH)
xs = np.arange(3)
order = ['own_mRNA', 'spamosaic', 'totalvi', 'dgat', 'mlp', 'nichetrans', 'txcellnet_clean', 'txcellnet']
STY = {'txcellnet': dict(color=HERO, lw=2.0, ls='-', marker='D', ms=4.4, zorder=6),
       'txcellnet_clean': dict(color=HERO_LIGHT, lw=1.2, ls=(0, (3, 1.5)), marker='D', ms=3.4, zorder=5)}
for m in order:
    st = STY[m] if m in STY else dict(color=COL[m], lw=1.2, ls=':' if m == 'own_mRNA' else '-', marker=MK[m], ms=3.8, zorder=4)
    if m == 'own_mRNA': st['color'] = '#B5B5B5'
    ax.plot(xs, TE.loc[m].values, mec='white', mew=0.4, **st)
ax.plot([2], [TE.loc['txcellnet_ens6', 'cross-dataset']], marker='*', ms=9.0, color=HERO, mec='white', mew=0.4, ls='none', zorder=7)
ax.set_xticks(xs, [l for _, l in TIERS], linespacing=LS); ax.set_xlim(-0.1, 2.1)
ax.set_ylim(0.15, 0.85); ax.set_yticks([0.2, 0.4, 0.6, 0.8]); ax.set_ylabel('Spearman (22 sections)')
ax.grid(axis='y', color='#EEEEEE', lw=0.5, zorder=0); ax.set_axisbelow(True)
NAMES = {'txcellnet_ens6': '6-net ensemble', 'own_mRNA': 'Cognate mRNA'}
lab_items = sorted([(TE.loc[m, 'cross-dataset'], m) for m in order if m != 'txcellnet_clean']
                   + [(TE.loc['txcellnet_ens6', 'cross-dataset'], 'txcellnet_ens6')])
# label positions: at least one 10 pt line apart, inside the y range of the plot
dpu = (0.85 - 0.15) / PH                          # data units per inch
gap, lo = 0.158 * dpu, 0.15 + 0.08 * dpu
ys = [v for v, _ in lab_items]
for _ in range(500):
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            dd = (gap - (ys[i] - ys[i - 1])) / 2; ys[i] += dd; ys[i - 1] -= dd
    if ys[0] < lo: ys = [y + lo - ys[0] for y in ys]
for (v, m), yl in zip(lab_items, ys):
    ax.annotate(f'{NAMES.get(m, LAB.get(m, m))}  {v:.2f}', xy=(2, v), xytext=(2.22, yl), textcoords='data', fontsize=FS,
                va='center', color=INK, fontweight='bold' if m == 'txcellnet' else 'normal', annotation_clip=False,
                arrowprops=dict(arrowstyle='-', color='#C8C8C8', lw=0.5, shrinkA=0.5, shrinkB=1.5, relpos=(0, 0.5)))

D = pd.DataFrame([dict(method=m, wins=w[0], sections=w[1], p_one_sided=pvals[m]) for m, w in wins.items()]).set_index('method')
save(fig, OUT, {'a_kidney_only': TC.round(4), 'b_paired': D, 'c_three_settings': TE.round(4)},
     panels={'a': axA, 'b': axB, 'c': axC}, row_groups=[['a', 'b', 'c']],
     exemptions=[{'panels': ['a', 'b', 'c'], 'checks': ['panel-width', 'horizontal-gutter'],
                  'reason': 'b needs room for four 10 pt method names, c for direct labels on the right; a carries its group '
                            'names above the rows; widths follow content'}])
pd.set_option('display.width', 200)
print(TC.round(3).to_string()); print(D.to_string()); print(TE.round(3).to_string())
