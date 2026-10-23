"""Supplementary Fig. S5: group means of every compared method in the three settings: cell = mean over the sections (pairs) of the group of the median cell-level Spearman over
markers, shaded by its rank within the column (darker = better), best per column in bold; last column = overall mean as a bar.
Every printed glyph >= 10 pt (figstyle.print_min10; panel letters 11 pt). To fit at 10 pt: the panel title sits in the free
header cell above the method names (short setting name as in main-text Fig. 2c, number of sections or pairs); the column
headers name the dataset group only (readouts as in Table 1, group names as in main-text Fig. 2a), and b groups its columns
under spanning dataset headers.
usage: figS5_tables.py DATA_DIR OUT_PREFIX
"""
import sys
import pandas as pd, matplotlib.pyplot as plt
import figstyle
figstyle.print_min10()
from figstyle import W, label, axes_in, save, INK, LS
from benchtable import bench_table, table_height, NAME_X

DD, OUT = sys.argv[1:3]
FS = 10
KEY = {'txcellnet': 'txcellnet_cr', 'nichetrans': 'nichetrans_off'}
PW = pd.read_csv(f'{DD}/tables/fig_within_per_dataset.csv', index_col=0)
PC = pd.read_csv(f'{DD}/tables/fig_cross_per_dataset.csv', index_col=0)
BREAST = [f'xen_breast_S{s}_{r}' for s in (1, 2, 3, 4) for r in ('Top', 'Mid', 'Bot')]
# (column key = source-data column name, printed header, sections)
GA = [('Kidney\nIF (2)', 'Kidney\n(2)', ['kidney_rcc', 'kidney_ccrcc']),
      ('CRLM\nIMC (6)', 'CRLM\n(6)', ['crlm_tumor1', 'crlm_tumor2', 'crlm_adjnormal', 'crlm_tumor3_A1', 'crlm_tumor3_A2', 'crlm_tumor3_A3']),
      ('Liver, tonsil\nCOMET (4)', 'Liver,\ntonsil (4)', ['duchini_core01_hcc', 'duchini_core02_non_tumour', 'duchini_core03_tonsil', 'duchini_core04_hca']),
      ('Breast\nstains (12)', 'Breast\n(12)', BREAST), ('Lung, panc.\nstains (2)', 'Lung,\npanc. (2)', ['xen_lung_v1', 'xen_pancreas_v1']),
      ('LN, fresh\nstains (8)', 'LN,\nfresh (8)', ['xen_lymphnode_5k', 'xen_skin_5k', 'xen_prostate_5k', 'xen_ovary_5k', 'xen_ovary_v1ff',
                                                  'xen_breast_5k', 'xen_cervix_5k', 'xen_gbm_v1'])]
TA = pd.DataFrame({h: PW.loc[cs].mean() for h, _, cs in GA}); TA['All'] = PW.mean()
XS = pd.read_csv(f'{DD}/final/summary/xsec_groups.csv'); XS = XS[XS.tier == 'cross-section']
XD = pd.read_csv(f'{DD}/final/summary/xsec_datasets.csv'); XD = XD[XD.tier == 'cross-section']
GB = [('Breast\nsame pt. (12)', 'Same\npt. (12)', 'breast, same patient'), ('Breast\nother pt. (12)', 'Other\npt. (12)', 'breast, other patient'),
      ('CRLM T3\nROIs (6)', 'T3\nROIs (6)', 'CRLM tumour 3, other ROI'), ('CRLM\nsections (6)', 'Sections\n(6)', 'CRLM, other section'),
      ('COMET\nliver (6)', 'Liver\n(6)', 'COMET, other liver core'), ('COMET\nliv./tonsil (6)', 'Liver–\ntonsil (6)', 'COMET, liver <-> tonsil'),
      ('Kidney\nRCC/ccRCC (2)', 'Other\npt. (2)', 'kidney, other patient')]
GB_SPAN = [('Breast', 2), ('CRLM', 2), ('COMET', 2), ('Kidney', 1)]
TB = pd.DataFrame({h: {m: XS[(XS.group == g) & (XS.method == KEY.get(m, m))].spearman.mean() for m in TA.index} for h, _, g in GB})
TB['All'] = {m: XD[XD.method == KEY.get(m, m)].groupby(['train', 'test']).spearman.mean().mean() for m in TA.index}
GC = [('Breast\nstains (9)', 'Breast\n(9)', [c for c in PC.index if c.startswith('xen_breast_S')]),
      ('Lymph node\nstains (1)', 'Lymph node\n(1)', ['xen_lymphnode_5k']),
      ('CRLM\nIMC (4)', 'CRLM\n(4)', [c for c in PC.index if c.startswith('crlm')]),
      ('Liver, tonsil\nCOMET (3)', 'Liver, tonsil\n(3)', [c for c in PC.index if c.startswith('duchini')]),
      ('Fresh\nstains (7)', 'Fresh\n(7)', ['xen_skin_5k', 'xen_prostate_5k', 'xen_ovary_5k', 'xen_ovary_v1ff', 'xen_breast_5k', 'xen_cervix_5k', 'xen_gbm_v1'])]
assert sum(len(c) for _, _, c in GC) == len(PC) == 24 and len(PW) == 34
TC = pd.DataFrame({h: PC.loc[cs].mean() for h, _, cs in GC}); TC['All'] = PC.mean()

ROW_H, GAP, SEP, TOP = 0.165, 0.3, 0.10, 0.03  # row pitch (in), gap between method blocks (x pitch), space between panels, top margin (in)
SPEC = [('a', 'Same section\n(34 sections)', TA, GA, None, 'All sections\n(mean)'),
        ('b', 'Other section\n(50 pairs)', TB, GB, GB_SPAN, 'All pairs\n(mean)'),
        ('c', 'Kidney only\n(24 sections)', TC, GC, None, 'All sections\n(mean)')]
probe = plt.figure(figsize=(W, 8))
hs = [table_height(probe, T, [d for _, d, _ in G], head, span, ROW_H, GAP, FS) for _, _, T, G, span, head in SPEC]
plt.close(probe)
H = round(TOP + sum(hs) + SEP * (len(hs) - 1) + 0.02, 2)
fig = plt.figure(figsize=(W, H))
axes, y = {}, TOP
for (pid, title, T, G, span, head), h in zip(SPEC, hs):
    label(fig, 0.0, y, pid)
    fig.text((NAME_X + 0.01) / W, 1 - (y + 0.01) / H, title, fontsize=FS, va='top', ha='left', color=INK, linespacing=LS)
    axes[pid] = ax = axes_in(fig, 0.0, y, W, h)
    bench_table(ax, T, [k for k, _, _ in G], [d for _, d, _ in G], overall_head=head, groups=span, row_h=ROW_H, gap=GAP, fs=FS)
    y += h + SEP
print(f'figure {W} x {H} in')
save(fig, OUT, {'a_same_section': TA.round(4), 'b_other_section': TB.round(4), 'c_kidney_only': TC.round(4)}, panels=axes,
     column_groups=[['a', 'b', 'c']])
