"""Figure 2: accuracy within a dataset and after transfer, per protein and per group of sections. Cell-level Spearman throughout.
(a) the three settings on identical test cells (held-out stripes of the 22 breast, CRLM and COMET sections), within-dataset
    settings first: trained on the same section, on another section of the same dataset, or on kidney only (slopegraph,
    direct labels). TxCellNet = clean + randomized network in every setting (dashed: clean network only). The star is the
    six-network TxCellNet ensemble used in the CRLM block of b and in c, labelled "TxCellNet ensemble".
(b) per protein (rows) and method (columns) on two datasets, the 21 proteins measured in both (all 27 / 22 proteins with
    values: Supplementary Table S13, table_per_protein.py; `fig2_bench.py DATA_DIR OUT all` draws every protein):
    kidney held-out stripes, same-section setting (the models of Supplementary Table S4: TxCellNet one clean network,
    NicheTrans and MLP single models; mean of the two sections);
    CRLM IMC, kidney-trained (the models of Supplementary Table S3: six-network TxCellNet ensemble, NicheTrans and MLP
    three-seed ensembles; median of the 4 test sections, as in Supplementary Fig. S2).
    Colour only (numbers are not legible at 10 pt in cells this size); outlined cell = best method for that protein;
    short dash = no cognate transcript (or, with `all`, protein not measured). Proteins grouped by category as in
    Supplementary Fig. S2.
(c) kidney-trained, 24 test and fresh sections (models of Supplementary Table S3): mean per group of sections (Cleveland
    dot plot), last row all 24.
The per-section paired differences are Supplementary Fig. S6 (figS6_paired.py).
Inputs (DATA_DIR): tables/fig_cross_per_dataset.csv, tables/fig_within_per_dataset.csv, final/summary/xsec_datasets.csv,
per_protein/bench_rows_kidney_crlm.csv (rows of results/bench/*/bench_rows.csv for the kidney and CRLM sections;
scripts/figdata/per_protein_rows.py); per-protein values from perprotein.py.
Every printed glyph >= 10 pt (figstyle.print_min10); panel letters 11 pt.
usage: fig2_bench.py DATA_DIR OUT_PREFIX [all]
"""
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import figstyle
figstyle.print_min10()
from figstyle import W, LS, label, axes_in, save, INK, COL, LAB, MK, HERO, HERO_LIGHT
from benchtable import SHADE
from perprotein import METHODS as HM, CAT, PRETTY, per_protein

DD, OUT = sys.argv[1:3]
ALL = len(sys.argv) > 3 and sys.argv[3] == 'all'
SHOWN = ['txcellnet', 'nichetrans', 'dgat', 'spamosaic', 'totalvi', 'mlp']
KEY = {'txcellnet': 'txcellnet_cr', 'nichetrans': 'nichetrans_off'}
FS = 10          # every text item (rcParams already 10 after print_min10)
Y0 = 0.03                                          # top margin: the 11 pt panel letters stay on the page
# MLP and NicheTrans share the circle marker and differ in colour only (navy vs dark grey): MLP gets open circles here
MLP_MK = dict(mfc='white', mec=COL['mlp'], mew=1.0)

# =============== b data: per protein and method, two datasets (layout first: it sets the figure height) ===============
PK, PCR = per_protein(DD)
if ALL:
    cats = [c for _, c in CAT]
else:                                              # proteins measured in both datasets
    both = set(PK.dropna(how='all').index) & set(PCR.dropna(how='all').index)
    cats = [[p for p in c if p in both] for _, c in CAT]
    assert sum(map(len, cats)) == 21
PROTS = [p for c in cats for p in c]
PK, PCR = PK.reindex(PROTS), PCR.reindex(PROTS)
CW, RH, CG = 0.165, 0.15, 0.05                     # cell width, row height, gap between protein categories (in)
GB = 0.10 / CW                                     # gap between the two datasets (cells)
NB = len(HM)
yrow, y = [], 0.0
for c in cats:
    for _ in c: yrow.append(y); y += 1
    y += CG / RH
y_end = y - CG / RH
YH = Y0 + 1.10                                     # top of the first heatmap row (in)
H = round(YH + y_end * RH + 0.02, 2)

fig = plt.figure(figsize=(W, H))
rend = fig.canvas.get_renderer()
tw = lambda t: t.get_window_extent(rend).width / fig.dpi          # printed width (in) of a text item
# left column: a above c; the plot heights share what the heatmap height leaves
PHC = 1.10 + 0.4 * (H - 4.45)
PHA = H - 1.92 - PHC


def title(x, y, s):
    fig.text(x / W, 1 - (y + 0.01) / H, s, fontsize=FS, va='top', color=INK)


# =============== a: three settings on identical test cells (within-dataset settings first) ===============
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

XA, YA, WA = 0.50, Y0 + 0.33, 1.30                 # plot left, top, width (in); height PHA
YLO, YHI = 0.15, 0.85
label(fig, 0.0, Y0, 'a'); title(0.17, Y0, 'Three settings, same test cells')
ax = axA = axes_in(fig, XA, YA, WA, PHA)
xs = np.arange(3)
order = ['own_mRNA', 'spamosaic', 'totalvi', 'dgat', 'mlp', 'nichetrans', 'txcellnet_clean', 'txcellnet']
# blue family and grey kept; dash patterns and markers tell the lines apart (MLP: long dashes, open circles)
DASH = {'nichetrans': '-', 'dgat': (0, (4, 1.6)), 'spamosaic': (0, (5, 1.4, 1.2, 1.4)), 'totalvi': (0, (1.3, 1.3)),
        'mlp': (0, (8, 2.2)), 'own_mRNA': ':'}
STY = {'txcellnet': dict(color=HERO, lw=2.0, ls='-', marker='D', ms=4.4, zorder=6),
       'txcellnet_clean': dict(color=HERO_LIGHT, lw=1.3, ls=(0, (3, 1.5)), marker='D', ms=3.4, zorder=5)}
for m in order:
    st = STY[m] if m in STY else dict(color=COL[m], lw=1.3, ls=DASH[m], marker=MK[m], ms=4.0, zorder=4)
    if m == 'own_mRNA': st['color'] = '#B5B5B5'
    st = dict(mec='white', mew=0.4, **st) if m != 'mlp' else dict(**MLP_MK, **st)
    ax.plot(xs, TE.loc[m].values, **st)
ax.plot([2], [TE.loc['txcellnet_ens6', 'cross-dataset']], marker='*', ms=10.0, color=HERO, mec='white', mew=0.4, ls='none', zorder=7)
ax.set_xticks(xs, [l for _, l in TIERS], linespacing=LS); ax.set_xlim(-0.1, 2.1)
ax.set_ylim(YLO, YHI); ax.set_yticks([0.2, 0.4, 0.6, 0.8]); ax.set_ylabel('Spearman (22 sections)')
# the star: its own label in the free upper right of the plot, with a leader line
ens = TE.loc['txcellnet_ens6', 'cross-dataset']
ens_lab = ax.annotate('TxCellNet\nensemble', xy=(2, ens), xytext=(1.86, 0.765), textcoords='data', fontsize=FS, ha='right',
                      va='center', linespacing=LS, color=HERO, fontweight='bold', zorder=8,
                      arrowprops=dict(arrowstyle='-', color=HERO, lw=0.7, shrinkA=1.0, shrinkB=4.5, relpos=(0.80, 0)))
# light grid lines; a line that would cross the star's label stops 3 pt before it
fig.canvas.draw()
bb = Text.get_window_extent(ens_lab, rend).transformed(ax.transData.inverted())    # the text only, not the leader
x3, y3 = 3 / 72 * 2.2 / WA, 3 / 72 * (YHI - YLO) / PHA                              # 3 pt in data units
for g in [0.2, 0.4, 0.6, 0.8]:
    x1 = bb.x0 - x3 if bb.y0 - y3 < g < bb.y1 + y3 else 2.1
    ax.plot([-0.1, x1], [g, g], color='#EEEEEE', lw=0.5, zorder=0, solid_capstyle='butt')
NAMES = {'txcellnet_clean': 'TxCellNet, clean', 'own_mRNA': 'Cognate mRNA'}
lab_items = sorted([(TE.loc[m, 'cross-dataset'], m) for m in order])
# label positions: neighbouring labels at least one 10 pt line (0.158 in) apart, inside the y range of the plot
dpu = (YHI - YLO) / PHA                              # data units per inch
gap, lo = 0.158 * dpu, YLO + 0.08 * dpu
ys = [v for v, _ in lab_items]
for _ in range(500):
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            dd = (gap - (ys[i] - ys[i - 1])) / 2; ys[i] += dd; ys[i - 1] -= dd
    if ys[0] < lo: ys = [y + lo - ys[0] for y in ys]
assert ys[-1] + gap / 2 < YHI
for (v, m), yl in zip(lab_items, ys):
    ax.annotate(NAMES.get(m, LAB.get(m, m)), xy=(2, v), xytext=(2.25, yl), textcoords='data', fontsize=FS, va='center',
                color=HERO if m.startswith('txcellnet') else INK, fontweight='bold' if m == 'txcellnet' else 'normal',
                annotation_clip=False,
                arrowprops=dict(arrowstyle='-', color='#C8C8C8', lw=0.5, shrinkA=0.5, shrinkB=2.0, relpos=(0, 0.5)))

# =============== b: per protein and method, two datasets ===============
XH = W - 0.02 - (2 * NB + GB) * CW                 # heatmap left edge (in)
XB0 = XH - 0.90                                    # panel letter b (row names take 0.88 in to the left of the heatmap)
label(fig, XB0, Y0, 'b'); title(XB0 + 0.17, Y0, 'Per protein')
axB = axes_in(fig, XH, YH, (2 * NB + GB) * CW, y_end * RH)
axB.set_xlim(0, 2 * NB + GB); axB.set_ylim(y_end, 0); axB.axis('off')
BLOCKS = [(0.0, PK, 'Kidney\nsame section'), (NB + GB, PCR, 'CRLM\nkidney-trained')]
for x0, P, head in BLOCKS:
    for i, p in enumerate(PROTS):
        yc = yrow[i] + 0.5
        v = P.loc[p].astype(float)
        best = v.idxmax() if v.notna().any() else None
        for j, m in enumerate(HM):
            if not np.isfinite(v[m]):           # no cognate transcript (or protein not measured): short grey dash
                axB.plot([x0 + j + 0.3, x0 + j + 0.7], [yc, yc], color='#B0B0B0', lw=0.8, solid_capstyle='butt')
                continue
            axB.add_patch(Rectangle((x0 + j + 0.04, yc - 0.44), 0.92, 0.88, fc=SHADE(np.clip(v[m], 0, 0.9) / 0.9), ec='none'))
            if m == best:
                axB.add_patch(Rectangle((x0 + j + 0.04, yc - 0.44), 0.92, 0.88, fc='none', ec=INK, lw=1.1, zorder=5))
    for j, m in enumerate(HM):
        axB.annotate({'own_mRNA': 'mRNA'}.get(m, LAB[m]), xy=(x0 + j + 0.5, 0), xycoords='data', xytext=(0, 3),
                     textcoords='offset points', rotation=90, rotation_mode='anchor', ha='left', va='center', fontsize=FS,
                     annotation_clip=False, color=HERO if m == 'txcellnet' else INK,
                     fontweight='bold' if m == 'txcellnet' else 'normal')
    fig.text((XH + (x0 + NB / 2) * CW) / W, 1 - (Y0 + 0.02) / H, head, fontsize=FS, ha='center', va='top', color=INK, linespacing=LS)
for i, p in enumerate(PROTS):
    axB.annotate(PRETTY.get(p, p), xy=(0, yrow[i] + 0.5), xycoords='data', xytext=(-3, 0), textcoords='offset points',
                 ha='right', va='center_baseline', fontsize=FS, color=INK, annotation_clip=False)
# colour key in the empty corner above the protein names
cax = axes_in(fig, XB0 + 0.10, Y0 + 0.62, 0.66, 0.08)
cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=SHADE, aspect='auto', extent=[0, 0.9, 0, 1])
cax.set_yticks([]); cax.set_xticks([0, 0.4, 0.8], ['0', '0.4', '0.8']); cax.tick_params(axis='x', length=2, pad=1.5)
for s in cax.spines.values(): s.set_visible(False)
cax.set_title('Spearman', fontsize=FS, pad=3, color=INK)

# =============== c: kidney-trained, group means over 24 sections ===============
PC = pd.read_csv(f'{DD}/tables/fig_cross_per_dataset.csv', index_col=0)           # 24 sections, kidney-trained
assert len(PC) == 24
GC = [('Breast (9)', [c for c in PC.index if c.startswith('xen_breast_S')]), ('Lymph node (1)', ['xen_lymphnode_5k']),
      ('CRLM (4)', [c for c in PC.index if c.startswith('crlm')]), ('Liver, tonsil (3)', [c for c in PC.index if c.startswith('duchini')]),
      ('Fresh (7)', ['xen_skin_5k', 'xen_prostate_5k', 'xen_ovary_5k', 'xen_ovary_v1ff', 'xen_breast_5k', 'xen_cervix_5k', 'xen_gbm_v1'])]
assert sum(len(c) for _, c in GC) == 24
TC = pd.DataFrame({h: PC.loc[cs].mean() for h, cs in GC}); TC['All (24)'] = PC.mean()
BASE = ['mlp', 'ridge', 'knn', 'celltype', 'own_mRNA']
assert TC.loc[BASE, 'All (24)'].astype(float).idxmax() == 'mlp'                    # MLP is the strongest baseline here

YC0 = YA + PHA + 0.50                              # c: panel-letter top (below a's two-line tick labels)
label(fig, 0.0, YC0, 'c'); title(0.17, YC0, 'Kidney-trained, by group of sections')
MS_HERO = 5.2
kx, ky = 0.02, YC0 + 0.32                          # method key, two lines
for n, m in enumerate(SHOWN):
    if n == 3: kx, ky = 0.02, ky + 0.19
    mk = dict(**MLP_MK) if m == 'mlp' else dict(mfc=COL[m], mec='white', mew=0.4)
    fig.add_artist(Line2D([(kx + 0.05) / W], [1 - ky / H], marker=MK[m], ms=MS_HERO if m == 'txcellnet' else (4.4 if m == 'mlp' else 5.0),
                          ls='none', **mk))
    t = fig.text((kx + 0.14) / W, 1 - ky / H, LAB[m], fontsize=FS, va='center', color=INK,
                 fontweight='bold' if m == 'txcellnet' else 'normal')
    kx += 0.14 + tw(t) + 0.22
XC, YCP, WC = XA, YC0 + 0.64, WA                   # same left edge and width as a; group names to the right, like a's labels
axC = axes_in(fig, XC, YCP, WC, PHC)
rows = list(TC.columns); yy = np.arange(len(rows), dtype=float); yy[-1] += 0.3
for y_, r in zip(yy, rows):
    v = TC.loc[SHOWN, r].astype(float)
    axC.plot([v.min(), v.max()], [y_, y_], color='#D0D0D0', lw=1.0, zorder=1, solid_capstyle='round')
    for m in SHOWN[::-1]:
        mk = dict(**MLP_MK) if m == 'mlp' else dict(mfc=COL[m], mec='white', mew=0.4)
        axC.plot(TC.loc[m, r], y_, marker=MK[m], ms=MS_HERO if m == 'txcellnet' else (4.0 if m == 'mlp' else 4.6), ls='none',
                 zorder=6 if m == 'txcellnet' else 3, **mk)
axC.set_yticks(yy, rows); axC.yaxis.tick_right()
axC.tick_params(axis='y', length=0, pad=6, labelright=True, labelleft=False)
axC.get_yticklabels()[-1].set_fontweight('bold')
axC.axhline((yy[-2] + yy[-1]) / 2, color='#BDBDBD', lw=0.6)
axC.set_ylim(yy[-1] + 0.55, -0.55); axC.spines['left'].set_visible(False); axC.spines['right'].set_visible(False)
axC.set_xlim(0.12, 0.66); axC.set_xticks([0.2, 0.4, 0.6]); axC.set_xlabel('Spearman (mean)')
axC.grid(axis='x', color='#EEEEEE', lw=0.5); axC.set_axisbelow(True)
# everything inside the page: c's x label ends above the bottom edge
fig.canvas.draw()
assert axC.xaxis.label.get_window_extent(rend).y0 / fig.dpi > 0.005, 'panel c runs off the bottom of the figure'

save(fig, OUT, {'a_three_settings': TE.round(4), 'b_per_protein_kidney': PK.round(4), 'b_per_protein_crlm': PCR.round(4),
                'c_kidney_only': TC.round(4)},
     panels={'a': axA, 'b': axB, 'c': axC}, column_groups=[['a', 'c']])   # b spans the right column; a and c stack on the left
pd.set_option('display.width', 200)
print(f'figure {W} x {H} in; panel b: {len(PROTS)} proteins')
print(TE.round(3).to_string()); print(PK.round(2).to_string()); print(PCR.round(2).to_string()); print(TC.round(3).to_string())
print('best per protein, kidney:', PK.dropna(how='all').idxmax(1).value_counts().to_dict(), ' CRLM:', PCR.dropna(how='all').idxmax(1).value_counts().to_dict())
