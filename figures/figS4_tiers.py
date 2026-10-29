"""Supplementary Fig. S4: the three evaluation settings per group of sections, on identical test cells
(held-out stripes of the test section). Input: xsec_groups.csv of scripts/evaluation/xsec_summary.py (mean over
test sections or pairs of the per-section median Spearman). TxCellNet: clean + randomized single networks (solid), clean
network alone (dashed), six-network ensemble of the main text (star). Kidney: single-section models, no cross-dataset setting.
Layout (10 pt): 3 x 3 grid, seven groups in reading order, the key in the two free cells next to the kidney panel; line and
marker sizes as in main-text Fig. 2c (markers not clipped at the top of the y range). Every printed glyph >= 10 pt
(figstyle.print_min10). Same numbers as the 5-7 pt version.
usage: figS4_tiers.py GROUPS_CSV OUT_PREFIX
"""
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle
figstyle.print_min10()
from figstyle import W, LS, axes_in, save, INK, COL, LAB, MK, HERO, HERO_LIGHT
FS = 10  # every text in this figure

G = pd.read_csv(sys.argv[1]); OUT = sys.argv[2]
GROUPS = [('breast, same patient', 'Breast: same patient,\nother region (12 pairs)'), ('breast, other patient', 'Breast: other patient,\nsame region (12)'),
          ('CRLM tumour 3, other ROI', 'CRLM tumor 3:\nother ROI (6)'), ('CRLM, other section', 'CRLM: other section,\n8–469 counts per cell (6)'),
          ('COMET, other liver core', 'COMET: other\nliver core (6)'), ('COMET, liver <-> tonsil', 'COMET: liver and\ntonsil cores (6)'),
          ('kidney, other patient', 'Kidney: RCC and ccRCC\n(2, single-section models)')]
LINES = [('own_mRNA', 'Cognate mRNA'), ('totalvi', None), ('spamosaic', None), ('dgat', None), ('mlp', None),
         ('nichetrans_off', 'NicheTrans'), ('txcellnet', 'TxCellNet, clean only'), ('txcellnet_cr', 'TxCellNet')]
STY = {'txcellnet_cr': dict(color=HERO, lw=2.0, ls='-', marker='D', ms=4.4, zorder=6),
       'txcellnet': dict(color=HERO_LIGHT, lw=1.2, ls=(0, (3, 1.5)), marker='D', ms=3.4, zorder=5),
       'own_mRNA': dict(color='#B5B5B5', lw=1.2, ls=':', marker='o', ms=3.4, zorder=3)}
MEW = 0.4
TIERS = ['within', 'cross-section', 'cross-dataset']; XL = ['Same\nsection', 'Other\nsection', 'Other\ndataset']


def style(m):
    key = 'nichetrans' if m == 'nichetrans_off' else m
    return key, (STY[m] if m in STY else dict(color=COL[key], lw=1.2, ls='-', marker=MK[key], ms=3.8, zorder=4))


# ---------------- geometry (inches) ----------------
NC = 3
PW, PH = 1.62, 1.22          # plot area of each group
X0, GX = 0.52, 0.48          # left edge of the first plot area (y label, tick labels); gap between plot areas (tick labels)
Y0 = 0.44                    # top of the first plot area (two-line 10 pt title above it)
GY = 0.92                    # bottom of one plot area to the top of the next (two-line tick labels, two-line title)
H = Y0 + 3 * PH + 2 * GY + 0.40
fig = plt.figure(figsize=(W, H))
out, PAN = {}, {}
for i, (g, title) in enumerate(GROUPS):
    r, c = divmod(i, NC)
    ax = PAN[f'g{i + 1}'] = axes_in(fig, X0 + c * (PW + GX), Y0 + r * (PH + GY), PW, PH)
    S = G[G.group == g]
    tiers = ['within (single section)', 'cross-section'] if g.startswith('kidney') else TIERS
    for m, _ in LINES:
        y = [S[(S.tier == t) & (S.method == m)].spearman.mean() if len(S[(S.tier == t) & (S.method == m)]) else np.nan for t in tiers]
        ax.plot(range(len(tiers)), y, mec='white', mew=MEW, clip_on=False, **style(m)[1]); out[(g, m)] = y + [np.nan] * (3 - len(y))
    e = S[(S.tier == 'cross-dataset') & (S.method == 'txcellnet_ens6')].spearman
    if len(e): ax.plot([2], [e.mean()], marker='*', ms=9.0, color=HERO, mec='white', mew=MEW, ls='none', zorder=7, clip_on=False)
    ax.set_xticks(range(len(tiers)), XL[:len(tiers)], fontsize=FS, linespacing=LS); ax.set_xlim(-0.3, len(tiers) - 0.7)
    ax.tick_params(axis='x', length=0, pad=3)
    ax.set_ylim(-0.05, 0.85); ax.set_yticks([0, 0.4, 0.8]); ax.grid(axis='y', color='#EEEEEE', lw=0.5); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=FS, pad=4, linespacing=LS, color=INK, loc='left')
    if c == 0: ax.set_ylabel('Spearman', fontsize=FS)
# key in the two free cells of the last row, its top level with the kidney title
hs = []
for m, lab in LINES[::-1]:
    key, st = style(m)
    hs.append(Line2D([], [], mec='white', mew=MEW, label=lab or LAB[key], **{k: v for k, v in st.items() if k != 'zorder'}))
hs.insert(1, Line2D([], [], marker='*', ms=9.0, color=HERO, mec='white', mew=MEW, ls='none', label='TxCellNet ensemble'))
LX, LY = X0 + PW + GX - 0.05, Y0 + 2 * (PH + GY) - 0.30
fig.legend(handles=hs, loc='upper left', bbox_to_anchor=(LX / W, 1 - LY / H), borderaxespad=0, borderpad=0, ncol=2,
           fontsize=FS, handlelength=2.2, labelspacing=0.55, columnspacing=1.6)
D = pd.DataFrame(out, index=['same section', 'other section', 'other dataset']).T
save(fig, OUT, {'tiers': D.round(4)}, panels=PAN, row_groups=[['g1', 'g2', 'g3'], ['g4', 'g5', 'g6']],
     column_groups=[['g1', 'g4', 'g7'], ['g2', 'g5'], ['g3', 'g6']])
print(D.round(3).to_string())
