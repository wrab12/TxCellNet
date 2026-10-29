"""Figure 4 (v6, RECOMB 10-pt floor): semi-synthetic benchmark organised by compartment. Real kidney inputs (transcripts,
DAPI, cell and nucleus segmentation); each synthetic protein is placed in a compartment chosen by its rule (membrane,
nucleus, cytoplasm; or an epithelial membrane protein induced by immune neighbours within 15/40/80 um), so the benchmark
tests whether TxCellNet puts every protein where it belongs - real panels are mostly membrane proteins.
(a) one held-out window of section RCC, the same cells in every column: four synthetic proteins (immune membrane,
    proliferation nucleus, epithelial cytoplasm, membrane induced within 40 um), ground truth, TxCellNet stage 1 (clean +
    randomized networks; every image and number of this figure is stage 1, the conditional mean) and NicheTrans (per-cell
    prediction painted into its cells); faint lines: cell outlines
(b) localisation per synthetic protein (held-out cells of both sections, mean of the two sections): share of the cell
    signal that lies in the intended compartment for the ground truth (bar), TxCellNet and the range of the six per-cell methods
    (a uniform fill gives the compartment's area share weighted by the predicted intensity, close to its area share);
    within-cell Pearson of TxCellNet (0 for every per-cell method)
(c) cell-level Spearman per synthetic protein and its mean over the six proteins with all, 50 % or 25 % of the test
    molecules kept (cell-level methods retrained on the thinned inputs); shading = rank within the column, bold = best
Every printed glyph >= 10 pt (figstyle.print_min10); panel letters 11 pt. The cell-level and within-cell numbers are the
stored benchmark rows (sim_ens, sim_sota); the compartment shares are new (sim_loc/sim_loc_rows.csv, sim_loc.py).
usage: fig4_sim.py DATA_DIR OUT_PREFIX
"""
import glob, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from skimage.segmentation import find_boundaries
import figstyle
figstyle.print_min10()
from figstyle import W, label, axes_in, save, INK, COL, LAB, MK, HERO, HERO_LIGHT, NEUTRAL, LS
from benchtable import order_rows, SHADE
import imaging
from imaging import show, CMY
imaging.MODE.update(mode='rank', gamma=1.8)
FS = 10  # every text in this figure (panel letters: figstyle.PANEL_FS = 11)

DD, OUT = sys.argv[1:3]
CH = ['mem_immune', 'nuc_prolif', 'cyto_epith', 'niche_r15', 'niche_r40', 'niche_r80']
KEYS = {'TxCellNet:ens_cr': 'txcellnet', 'TxCellNet:sim_plain': 'txcellnet_clean', 'NicheTrans': 'nichetrans', 'DGAT': 'dgat',
        'SpaMosaic': 'spamosaic', 'totalVI': 'totalvi', 'ridge_niche': 'ridge_niche', 'ridge': 'ridge'}
D = pd.concat([pd.read_csv(f).query("method != 'NicheTrans'") for f in glob.glob(f'{DD}/sim_ens/sim_rows_*.csv')] +
              [pd.read_csv(f) for f in glob.glob(f'{DD}/sim_sota/sim_rows_*.csv')])
D = D[D.method.isin(KEYS)].assign(method=lambda d: d.method.map(KEYS))
base = D[D.perturb == 'none']
LOC = pd.read_csv(f'{DD}/sim_loc/sim_loc_rows.csv')
LOC = LOC.assign(method=LOC.method.map(lambda m: KEYS.get(m, m)))
CR = np.load(f'{DD}/sim_loc/crops_common_rcc.npz')
NAMES = {'txcellnet_clean': 'Clean only', 'ridge_niche': 'Ridge + nbhd'}  # nbhd = neighbourhood (defined in the caption)
name = lambda m: NAMES.get(m, LAB.get(m, m))
GREEN = np.array([0.30, 0.90, 0.35])
CCOL = {'mem_immune': CMY[1], 'nuc_prolif': CMY[2], 'cyto_epith': GREEN, 'niche_r15': CMY[0], 'niche_r40': CMY[0], 'niche_r80': CMY[0]}

H = 4.44
fig = plt.figure(figsize=(W, H)); REND = fig.canvas.get_renderer()
PAN = {}  # panel id -> axes, for the alignment audit


def txt(x, y, s, **kw):
    """figure text at (x, y) inches from the top-left corner, 10 pt"""
    kw.setdefault('fontsize', FS); kw.setdefault('color', INK)
    return fig.text(x / W, 1 - y / H, s, **kw)


def width_in(s, weight='normal'):
    t = fig.text(0, 0, s, fontsize=FS, fontweight=weight); w = t.get_window_extent(REND).width / fig.dpi; t.remove()
    return w


def square(x, y, c, s=0.085):
    """colour square, left edge x, centred on y (inches)"""
    fig.patches.append(Rectangle((x / W, 1 - (y + s / 2 + 0.004) / H), s / W, s / H, transform=fig.transFigure, fc=c, ec='none',
                                 figure=fig))


# ---------------- (a) synthetic proteins by compartment ----------------
T, G, X0, Y0 = 0.60, 0.045, 0.78, 0.47   # tiles: size, gap, left edge, top edge
WIN = 0                                   # window of crops_common_rcc.npz (rank 0: most cells expressing all four)
SHOW = [('mem_immune', 'Membrane'), ('nuc_prolif', 'Nucleus'), ('cyto_epith', 'Cytoplasm'), ('niche_r40', 'Induced')]
ROWS = [('gt', 'Truth', 'normal'), ('tx', ('TxCellNet', 'stage 1'), 'bold'), ('nichetrans', 'NicheTrans\n(per cell)', 'normal')]
CHW = [str(c) for c in CR['channels']]
lab = CR[f'w{WIN}_lab']; cellm = lab > 0; edge = find_boundaries(lab, mode='inner')
label(fig, 0.0, 0.02, 'a'); txt(0.17, 0.02, 'Synthetic proteins by compartment', va='top')
for c, (ch, head) in enumerate(SHOW):
    x = X0 + c * (T + G); j = CHW.index(ch)
    txt(x + T / 2, Y0 - 0.035, head, ha='center', va='bottom')
    for r, (k, rl, fw) in enumerate(ROWS):
        y = Y0 + r * (T + G); a = CR[f'w{WIN}_{k}'][j].astype(np.float32)
        u = imaging.unit(a, cellm)
        rgb = u[..., None] * CCOL[ch][None, None, :]
        rgb = np.maximum(rgb, edge[..., None] * 0.2); rgb[~cellm] = 0
        ax = PAN[f'a{r + 1}{c + 1}'] = axes_in(fig, x, y, T, T); show(ax, rgb)
        if c == 0 and r == 0: imaging.scalebar(ax, 20, text=False)   # length in the caption
        if c == 0 and isinstance(rl, tuple):   # bold method name over a plain second line, same pitch as a two-line label
            b1 = y + T / 2 - (LS * FS / 72 - 0.7 * FS / 72) / 2   # first baseline: block of cap height + one pitch centred
            txt(X0 - 0.06, b1, rl[0], ha='right', va='baseline', fontweight=fw)
            txt(X0 - 0.06, b1 + LS * FS / 72, rl[1], ha='right', va='baseline')
        elif c == 0: txt(X0 - 0.06, y + T / 2, rl, ha='right', va='center', fontweight=fw, linespacing=LS)
win_yx = CR[f'w{WIN}_yx']; print('a: window (y, x) px', win_yx.tolist(), 'size', lab.shape)

# ---------------- (b) localisation per synthetic protein ----------------
XB = 3.62                                   # panel left edge
RLX = XB + 0.02                             # row labels: colour square, then text
AXL, AXW = 4.80, 1.08                       # dot plot: left edge, width (in)
XWC = 6.25                                  # within-cell r column centre
RP, GAP, YR0 = 0.205, 0.06, 0.71            # row pitch, gap between the two groups, first row centre
RNAME = {'mem_immune': 'Membrane', 'nuc_prolif': 'Nucleus', 'cyto_epith': 'Cytoplasm', 'niche_r15': 'Induced, 15 µm',
         'niche_r40': 'Induced, 40 µm', 'niche_r80': 'Induced, 80 µm'}
yrow = {c: YR0 + i * RP + (GAP if i >= 3 else 0) for i, c in enumerate(CH)}
label(fig, XB, 0.02, 'b'); txt(XB + 0.17, 0.02, 'Localization per synthetic protein', va='top')
PERCELL = ['nichetrans', 'dgat', 'spamosaic', 'totalvi', 'ridge_niche', 'ridge']
S = LOC.groupby(['method', 'channel'])[['share_in', 'within_cell']].mean()
BTAB = pd.DataFrame({'truth_share': [S.loc[('truth', c), 'share_in'] for c in CH],
                     'txcellnet_share': [S.loc[('txcellnet', c), 'share_in'] for c in CH],
                     'percell_share_min': [min(S.loc[(m, c), 'share_in'] for m in PERCELL) for c in CH],
                     'percell_share_max': [max(S.loc[(m, c), 'share_in'] for m in PERCELL) for c in CH],
                     'txcellnet_within_cell': [base[(base.method == 'txcellnet') & (base.channel == c)].within_cell.mean() for c in CH]},
                    index=CH)
assert np.allclose(BTAB.txcellnet_within_cell, [S.loc[('txcellnet', c), 'within_cell'] for c in CH], atol=1e-4)
ytop, ybot = YR0 - RP / 2, yrow[CH[-1]] + RP / 2
ax = PAN['b'] = axes_in(fig, AXL, ytop, AXW, ybot - ytop)
ax.set_ylim(ybot - ytop, 0); ax.set_xlim(0.4, 1.0); ax.set_yticks([])
ax.set_xticks([0.4, 0.6, 0.8, 1.0], ['40', '60', '80', '100']); ax.spines['left'].set_visible(False)
ax.grid(axis='x', color='#EEEEEE', lw=0.5); ax.set_axisbelow(True)
ax.set_xlabel('Signal in compartment (%)', labelpad=2)
for c in CH:
    yy = yrow[c] - ytop; r = BTAB.loc[c]
    ax.plot([0.4, 1.0], [yy, yy], color='#F2F2F2', lw=0.5, zorder=0)
    ax.plot([r.percell_share_min, r.percell_share_max], [yy, yy], color='#BDBDBD', lw=4.5, solid_capstyle='round', zorder=2)
    ax.plot([r.truth_share], [yy], '|', ms=9, mec=INK, mew=1.6, zorder=3)
    ax.plot([r.txcellnet_share], [yy], 'D', ms=4.8, mfc=HERO, mec='white', mew=0.4, zorder=4)
    square(RLX, yrow[c], CCOL[c]); txt(RLX + 0.13, yrow[c], RNAME[c], va='center')
    v = BTAB.loc[c, 'txcellnet_within_cell']; txt(XWC, yrow[c], f'{v:.2f}', ha='center', va='center')
txt(XWC, YR0 - RP / 2 - 0.03, 'Within-\ncell r', ha='center', va='bottom', linespacing=LS)
# key (two lines above the dot plot, left of the within-cell header)
yk1, yk2 = 0.30, 0.47; xk = RLX
fig.add_artist(Line2D([(xk + 0.045) / W], [1 - yk1 / H], ls='none', marker='D', ms=4.8, mfc=HERO, mec='white', mew=0.4))
t = txt(xk + 0.13, yk1, 'TxCellNet', va='center', fontweight='bold'); xk2 = xk + 0.13 + width_in('TxCellNet', 'bold') + 0.2
fig.add_artist(Line2D([(xk2 + 0.045) / W], [1 - yk1 / H], ls='none', marker='|', ms=9, mec=INK, mew=1.6))
txt(xk2 + 0.13, yk1, 'Truth', va='center')
fig.add_artist(Line2D([(xk + 0.03) / W, (xk + 0.07) / W], [1 - yk2 / H] * 2, color='#BDBDBD', lw=4.5, solid_capstyle='round'))
txt(xk + 0.13, yk2, 'Per-cell methods (range)', va='center')
print('b: key ends at', round(xk + 0.13 + width_in('Per-cell methods (range)'), 3), 'in; within-cell header starts at',
      round(XWC - width_in('Within-') / 2, 3), 'in; row labels end at', round(RLX + 0.13 + width_in('Induced, 15 µm'), 3), 'in')

# ---------------- (c) cell-level Spearman table ----------------
yc = 2.50
R = D.groupby(['perturb', 'method']).cell_spearman.mean().unstack('method')
TBL = base.groupby(['method', 'channel']).cell_spearman.mean().unstack('channel')[CH]
for p, col in (('none', 'kept100'), ('thin50', 'kept50'), ('thin25', 'kept25')): TBL[col] = R.loc[p]
assert np.allclose(TBL.kept100, base.groupby('method').cell_spearman.mean().reindex(TBL.index))
cols_c = CH + ['kept100', 'kept50', 'kept25']
heads = ['Membrane', 'Nucleus', 'Cytoplasm', '15 µm', '40 µm', '80 µm', '100 %', '50 %', '25 %']
groups = [('Cell’s own program', 3), ('Induced by neighbors', 3), ('Mean of six, molecules kept', 3)]
fmt = lambda v: f'{v:.2f}'
BLOCKS = [['txcellnet', 'txcellnet_clean']] + [b for b in order_rows(TBL.drop(index='txcellnet_clean'), 'kept100')[1:] if b]


def table_c(yc):
    """10-pt table in printed inches: column pitch from the header widths, shading by rank within the column, bold best"""
    RPc, BGc = 0.17, 0.05                        # row pitch, gap between the blocks
    NAME_W, GG = 1.04, 0.07                      # name column, gap between groups
    hw = [width_in(h) for h in heads]; pitch, j0 = [], 0
    for gl, n in groups:
        p = max([0.46] + [(hw[j] + hw[j + 1]) / 2 + 0.05 for j in range(j0, j0 + n - 1)] + [(width_in(gl) + 0.1) / n])
        pitch += [p] * n; j0 += n
    xs, x, bounds = [], NAME_W, np.cumsum([n for _, n in groups])[:-1]
    for j in range(len(cols_c)):
        if j in bounds: x += GG
        xs.append(x); x += pitch[j]
    y_grp, y_hd = yc + 0.075, yc + 0.37          # group headers (centre), column headers (bottom)
    y0 = y_hd + 0.03 + RPc / 2
    axc = axes_in(fig, 0, yc - 0.02, W - 0.02, H - yc + 0.01); axc.set_xlim(0, W - 0.02); axc.set_ylim(H - yc + 0.01, 0)
    axc.axis('off'); Y = lambda y: y - (yc - 0.02)
    label(fig, 0.0, yc - 0.01, 'c')
    axc.text(0.17, Y(yc - 0.01), 'Cell-level\nSpearman', ha='left', va='top', fontsize=FS, color=INK, linespacing=LS)
    j0 = 0
    for gl, n in groups:
        xa_, xb_ = xs[j0] + 0.04, xs[j0 + n - 1] + pitch[j0 + n - 1] - 0.04
        axc.plot([xa_, xb_], [Y(y_grp + 0.09)] * 2, color='#9A9A9A', lw=0.6, solid_capstyle='butt')
        axc.text((xa_ + xb_) / 2, Y(y_grp), gl, ha='center', va='center', fontsize=FS, color=INK)
        j0 += n
    rows, y = [], y0
    for block in BLOCKS:
        for m in block: rows.append((m, y)); y += RPc
        y += BGc
    for j, c in enumerate(cols_c):
        v = TBL[c].astype(float).reindex([m for m, _ in rows])
        rk = v.rank(ascending=True, method='min'); rk = (rk - 1) / max(len(v) - 1, 1)  # 0 = worst, 1 = best
        best = v.max(); xc = xs[j] + pitch[j] / 2
        axc.text(xc, Y(y_hd), heads[j], ha='center', va='bottom', fontsize=FS, color=INK)
        for m, yy in rows:
            val = v[m]; colr = SHADE(0.08 + 0.92 * rk[m]); lum = np.dot(mcolors.to_rgb(colr), [0.299, 0.587, 0.114])
            axc.add_patch(Rectangle((xs[j] + 0.03, Y(yy) - 0.46 * RPc), pitch[j] - 0.06, 0.92 * RPc, fc=colr, ec='none'))
            axc.text(xc, Y(yy), fmt(val), ha='center', va='center', fontsize=FS, color='white' if lum < 0.55 else INK,
                     fontweight='bold' if fmt(val) == fmt(best) else 'normal')
    for m, yy in rows:
        st = dict(marker=MK['txcellnet'], ms=4.4, mfc=HERO_LIGHT) if m == 'txcellnet_clean' else dict(marker=MK[m], ms=5, mfc=COL[m])
        axc.plot([0.07], [Y(yy)], mec='white', mew=0.5, clip_on=False, **st)
        axc.text(0.15, Y(yy), name(m), ha='left', va='center', fontsize=FS, color=INK, fontweight='bold' if m == 'txcellnet' else 'normal')
    print(f'c: right edge of the last column {xs[-1] + pitch[-1]:.3f} in, last row bottom {rows[-1][1] + RPc / 2:.3f} in '
          f'(figure {W} x {H} in)')
    return axc


table_c(yc)
SRC = {'a_window_rcc': pd.DataFrame({'y_px': [int(win_yx[0])], 'x_px': [int(win_yx[1])], 'size_px': [lab.shape[0]],
                                     'um_per_px': [imaging.UM]}),
       'b_localization': BTAB.round(4),
       'c_cell_spearman': TBL.loc[[m for b in BLOCKS for m in b], cols_c].round(4)}
save(fig, OUT, SRC, panels=PAN, row_groups=[[f'a{r}{c}' for c in range(1, 5)] for r in range(1, 4)],
     column_groups=[[f'a{r}{c}' for r in range(1, 4)] for c in range(1, 5)])
print(BTAB.round(3).to_string()); print(SRC['c_cell_spearman'].round(3).to_string())
