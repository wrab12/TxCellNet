"""Figure 5: case study. Colorectal liver metastases with same-section IMC; every model trained on
kidney only. The same rule is applied to measured IMC and to each RNA-based source (per-section Otsu thresholds on PanCK, CD45,
CD68; tumour region = cells with at least half tumour cells within 25 um; signed distance of each cell to the region boundary).
(a) tumour 2, the shallowest tumour section: tumour region and CD45+ cells called from each source; below each map the Dice of
the tumour region and the Pearson r of the CD45+ profile (b) with IMC;
(b, c) fraction of CD45+ and CD68+ cells against the distance to the tumour boundary in tumour 2 (bands: 95% intervals,
spatial block bootstrap of 200-um tiles, for IMC and TxCellNet); one line key for b-d under the panel titles;
(d) agreement with IMC in each of the five tumour sections (dots: sections; tick: mean; black outline: tumour 2);
Dice = tumour-region Dice, CD45 r / CD68 r = Pearson r of the profiles in b / c with the IMC profile (as in a).
Every printed glyph >= 10 pt (figstyle.print_min10); panel letters 11 pt.
usage: fig5_case.py CASE_DIR OUT_PREFIX   (CASE_DIR: summary.csv, profiles.csv, map_<section>.npz of crlm_case.py)
"""
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import figstyle
figstyle.print_min10()
from figstyle import W, label, axes_in, save, INK, COL, HERO

CD, OUT = sys.argv[1:3]
S = pd.read_csv(f'{CD}/summary.csv'); P = pd.read_csv(f'{CD}/profiles.csv')
SHOW = 'crlm_tumor2'
SECTIONS = ['crlm_tumor1', 'crlm_tumor2', 'crlm_tumor3_A1', 'crlm_tumor3_A2', 'crlm_tumor3_A3']
SRC = [('IMC', 'IMC (measured)'), ('TxCellNet', 'TxCellNet'), ('DGAT', 'DGAT'), ('NicheTrans', 'NicheTrans'),
       ('Cognate mRNA', 'Cognate mRNA')]
SCOL = {'IMC': INK, 'TxCellNet': HERO, 'DGAT': COL['dgat'], 'NicheTrans': COL['nichetrans'], 'Cognate mRNA': '#ABABAB'}
TUMOUR, IMMUNE, OTHER = '#9ADBF0', '#C2188F', '#DADADA'   # tints of the cyan / magenta of the image figures
FS = 10                                                    # every text in this figure (panel letters: figstyle.PANEL_FS = 11)
num = lambda v, f='.2f': format(v, f).replace('-', '−')   # typographic minus

H = 3.62
fig = plt.figure(figsize=(W, H))
R = fig.canvas.get_renderer()


def yy(y):
    return 1 - y / H


def txt(x, y, s, **kw):
    """text at (x, y) inches from the top-left corner"""
    kw.setdefault('fontsize', FS); kw.setdefault('color', INK)
    return fig.text(x / W, yy(y), s, **kw)


def width(s, **kw):
    """printed width (in) of a text before placing it"""
    t = txt(0, 0, s, **kw); w = t.get_window_extent(R).width / fig.dpi; t.remove(); return w


def title(x, y, s):
    """panel title on the line of the panel letter (baseline matched to the 11-pt letter)"""
    txt(x, y + 0.012, s, va='top')


# ---- a: maps of tumour 2 ----
label(fig, 0.0, 0.02, 'a'); title(0.20, 0.02, 'Tumour 2 (10 molecules per cell)')
SQ, KP, KS = 0.10, 0.05, 0.17   # colour key: square, square-to-text, entry spacing (in)
items = [(TUMOUR, 'Tumour region'), (IMMUNE, 'CD45+ cell'), (OTHER, 'Other cell')]
kx = 6.45 - sum(SQ + KP + width(s) for _, s in items) - KS * (len(items) - 1)   # right-aligned with the last map
ky = 0.02 + 0.085
for c, s in items:
    fig.patches.append(Rectangle((kx / W, yy(ky + SQ / 2)), SQ / W, SQ / H, transform=fig.transFigure, fc=c, ec='none',
                                 figure=fig))
    kx += SQ + KP + txt(kx + SQ + KP, ky, s, va='center').get_window_extent(R).width / fig.dpi + KS
z = np.load(f'{CD}/map_{SHOW}.npz'); xy = z['xy']
ctr, span = (xy.min(0) + xy.max(0)) / 2, (xy.max(0) - xy.min(0)).max() + 30   # square window = square axes
MW, MG, X0, Y0 = 1.10, 0.20, 0.10, 0.42
row = S[S.case == SHOW].set_index('source')
best_d = row.drop('IMC').dice_tumour.max(); best_r = row.drop('IMC').r_immune.max()
maps = {}
for j, (s, name) in enumerate(SRC):
    x = X0 + j * (MW + MG); cx = x + MW / 2
    ax = maps[f'a{j + 1}'] = axes_in(fig, x, Y0, MW, MW)
    tum, imm, reg = z[s.replace(' ', '_')].astype(bool)
    for sel, c in (((~reg) & (~imm), OTHER), (reg & (~imm), TUMOUR), (imm, IMMUNE)):
        ax.scatter(xy[sel, 0], xy[sel, 1], s=0.35, c=c, linewidths=0, rasterized=True)
    ax.set_xlim(ctr[0] - span / 2, ctr[0] + span / 2); ax.set_ylim(ctr[1] + span / 2, ctr[1] - span / 2); ax.axis('off')
    txt(cx, Y0 - 0.05, name, ha='center', va='baseline', fontweight='bold' if s == 'TxCellNet' else 'normal')
    ys = Y0 + MW + 0.13
    if s == 'IMC':   # scale bar
        x0d, x1d = ax.get_xlim(); bar = 500 / ((x1d - x0d) / MW)
        fig.add_artist(Line2D([(x + 0.04) / W, (x + 0.04 + bar) / W], [yy(ys)] * 2, color=INK, lw=1.2, solid_capstyle='butt'))
        txt(x + 0.04 + bar + 0.07, ys, '500 µm', va='center')
        continue
    d, r = row.loc[s, 'dice_tumour'], row.loc[s, 'r_immune']
    bd = 'bold' if f'{d:.2f}' == f'{best_d:.2f}' else 'normal'; br = 'bold' if f'{r:.2f}' == f'{best_r:.2f}' else 'normal'
    sd, sr, gap = f'Dice {num(d)}', f'r {num(r)}', 0.10
    x1 = cx - (width(sd, fontweight=bd) + gap + width(sr, fontweight=br)) / 2   # the pair centred under its map
    t = txt(x1, ys, sd, va='center', fontweight=bd)
    txt(x1 + t.get_window_extent(R).width / fig.dpi + gap, ys, sr, va='center', fontweight=br)

# ---- b, c: profiles across the tumour boundary (tumour 2) ----
Y2, PH = 1.82, 0.90
BX, BW, CX = 0.50, 1.24, 2.26                       # b, c: axes left edges and width (in)
label(fig, 0.0, Y2, 'b'); title(0.20, Y2, 'Immune cells')
label(fig, 1.80, Y2, 'c'); title(2.00, Y2, 'Macrophages')
kx, ky = BX, Y2 + 0.31                               # one line key for b-d (d: same colours)
for s, name in SRC:
    st = dict(color=SCOL[s], lw=1.4 if s in ('IMC', 'TxCellNet') else 1.0, ls=':' if s == 'Cognate mRNA' else '-')
    fig.add_artist(Line2D([kx / W, (kx + 0.20) / W], [yy(ky)] * 2, **st))
    t = txt(kx + 0.25, ky, name, va='center')
    kx += 0.25 + t.get_window_extent(R).width / fig.dpi + 0.17
YP = ky + 0.20
Q = P[P.case == SHOW]; NMIN, XMIN = 100, -92   # bins plotted: at least 100 cells, inside the x range
prof_axes = {}
for pid, (col, ylab, ylim, yt, x) in {'b': ('immune', 'CD45+ fraction', (0, 1.0), [0, 0.5, 1.0], BX),
                                      'c': ('macrophage', 'CD68+ fraction', (0, 0.6), [0, 0.3, 0.6], CX)}.items():
    ax = prof_axes[pid] = axes_in(fig, x, YP, BW, PH)
    ax.axvspan(XMIN, 0, color='#EAF6FB', lw=0, zorder=0)
    for s, _ in SRC[::-1]:
        q = Q[Q.source == s].sort_values('distance'); ok = q[col].notna() & (q.n >= NMIN) & (q.distance > XMIN)
        d, v = q.distance[ok], q[col][ok]
        if s in ('IMC', 'TxCellNet'):
            ax.fill_between(d, q[f'{col}_lo'][ok], q[f'{col}_hi'][ok], color=SCOL[s], alpha=0.18 if s == 'IMC' else 0.22, lw=0,
                            zorder=2)
            ax.plot(d, v, color=SCOL[s], lw=1.4, zorder=5 if s == 'IMC' else 6)
        else:
            ax.plot(d, v, color=SCOL[s], lw=1.0, ls=':' if s == 'Cognate mRNA' else '-', zorder=3)
    ax.axvline(0, color='#9A9A9A', lw=0.5, zorder=1)
    ax.set_xlim(XMIN, 150); ax.set_xticks([-80, 0, 80]); ax.set_ylim(*ylim); ax.set_yticks(yt); ax.set_ylabel(ylab)
xl = prof_axes['b'].set_xlabel('Distance to tumour boundary (µm)')   # one x label shared by b and c, centred under both
xl.set_x((CX + BW - BX) / 2 / BW)

# ---- d: agreement with IMC, five tumour sections ----
label(fig, 3.62, Y2, 'd'); title(3.82, Y2, 'Agreement with IMC, five sections')
METRICS = [('dice_tumour', 'Dice', (0, 1), [0, 0.5, 1]), ('r_immune', 'CD45 r', (-1, 1), [-1, 0, 1]),
           ('r_macrophage', 'CD68 r', (-1, 1), [-1, 0, 1])]
ROWS = ['TxCellNet', 'DGAT', 'NicheTrans', 'Cognate mRNA']
OFF = np.linspace(-0.24, 0.24, len(SECTIONS))
DW, DG, XD = 0.52, 0.12, 4.65
dax = {}
for k, (col, lab, lim, ticks) in enumerate(METRICS):
    ax = dax[f'd{k + 1}'] = axes_in(fig, XD + k * (DW + DG), YP, DW, PH)
    pad = 0.08 * (lim[1] - lim[0])
    for i, s in enumerate(ROWS):
        v = np.array([S[(S.case == c) & (S.source == s)][col].item() for c in SECTIONS])
        ax.plot([v.mean()] * 2, [i - 0.36, i + 0.36], color=SCOL[s], lw=1.2, zorder=2, solid_capstyle='butt')
        for j, c in enumerate(SECTIONS):
            ax.plot(v[j], i + OFF[j], marker='o', ms=3.0, ls='none', mfc=SCOL[s], zorder=4 if c == SHOW else 3,
                    mec=INK if c == SHOW else 'white', mew=0.6 if c == SHOW else 0.3)
    ax.set_ylim(len(ROWS) - 0.5, -0.5); ax.set_xlim(lim[0] - pad, lim[1] + pad); ax.set_xticks(ticks, [num(t, 'g') for t in ticks])
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(ROWS if k == 0 else []); ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False); ax.grid(axis='x', color='#EEEEEE', lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel(lab)
    if k == 0:
        for t in ax.get_yticklabels(): t.set_fontweight('bold' if t.get_text() == 'TxCellNet' else 'normal')

cols = ['case', 'source', 'median_counts', 'tumour_region', 'immune', 'dice_tumour', 'dice_immune', 'r_immune', 'r_macrophage',
        'immune_in', 'immune_out', 'exclusion']
counts = pd.DataFrame({s: {'tumour_region': z[s.replace(' ', '_')][2].mean(), 'cd45_cells': z[s.replace(' ', '_')][1].mean()}
                       for s, _ in SRC}).T
save(fig, OUT, {'a_tumour2_fractions': counts.round(4), 'bc_profiles_tumour2': Q.round(4).set_index(['source', 'distance']),
                'd_agreement': S[cols].round(4).set_index(['case', 'source'])},
     panels={**maps, **prof_axes, **dax}, row_groups=[list(maps), ['b', 'c'], list(dax)])
print(S[S.source != 'IMC'].groupby('source')[['dice_tumour', 'r_immune', 'r_macrophage']].mean().round(3))
