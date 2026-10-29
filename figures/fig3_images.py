"""Figure 3 (v5, RECOMB 10-pt floor; Fig. 4 in v3/v4): what an image adds over per-cell predictions, and what stage 2 adds.
(a) held-out kidney window as a three-protein composite on black (cyan E-cadherin: epithelial membrane; magenta CD45:
    immune-cell membrane; yellow Ki-67: nucleus of proliferating cells): measured | TxCellNet stage 1 | stage 2 | NicheTrans |
    DGAT | SpaMosaic | totalVI (per-cell predictions painted into their cells); second row: zoom (box); under the tiles a
    small table, one row per protein (colour square + name under 'Measured'), one column per method: cell-level Spearman of
    each protein over all held-out cells of the section (bold: best in the row).
(b) within-cell Pearson per protein (pixel minus cell mean), held-out kidney; painted per-cell predictions are 0.
(c) boundary-minus-interior enrichment per protein (z), TxCellNet stage 1 against measured, held-out kidney.
(d) stage-2 realism: per protein, distance(stage 2) / distance(stage 1) to the measured image on 11 image statistics and the
    two correlations, with the matched-noise control; same code and numbers as figS1b_stage2_realism.py (see its docstring
    for the metrics), drawn here as the bottom band.
Each channel of each image panel is scaled between its own median and 99.5th percentile over the panel's cell pixels
(display only). The colorectal liver metastasis window of the v5 Fig. 3b is in Supplementary Fig. S3b (figS3_allmethods.py).
Layout (10 pt): a full width; b (left) and c (right) share the middle band; d full width at the bottom. Every printed glyph
>= 10 pt (figstyle.print_min10), panel letters 11 pt. The protein colour key is the first column of the Spearman table;
the compartment of each kidney protein is given in the caption. Same numbers as the earlier versions of panels a-c.
usage: fig3_images.py DATA_DIR OUT_PREFIX [REALISM_CSV]
  REALISM_CSV: stage2_realism_per_protein_all.csv (default: DATA_DIR/stage2_realism_per_protein_all.csv)
"""
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from skimage.segmentation import find_boundaries
import figstyle
figstyle.print_min10()
from figstyle import W, LS, label, axes_in, save, INK, COL, HERO, HERO_LIGHT, NEUTRAL
from imaging import CMY, composite, show, UM

FS = 10  # every text in this figure (panel letters: figstyle.PANEL_FS = 11)
DD, OUT = sys.argv[1:3]
REAL = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DD, 'stage2_realism_per_protein_all.csv')
MET = pd.read_csv(f'{DD}/metrics_allm.csv')
rho = lambda case, m, p: MET[(MET.case == case) & (MET.method == m) & (MET.marker == p)].spearman.mean()
PRETTY = {'E-Cadherin': 'E-cadherin', 'Beta-catenin': 'β-catenin', 'alphaSMA': 'αSMA', 'GranzymeB': 'GzmB', 'CD8A': 'CD8',
          'CD3E': 'CD3'}

# ---------------- geometry (inches from the top-left corner) ----------------
X0, T, G, Y0 = 0.45, 0.78, 0.045, 0.45     # (a) image tiles: left edge, size, gap, top edge
XSP = 0.27                                  # centre of the rotated 'Spearman' label left of each table
RP = 0.16                                   # row pitch of the Spearman table
YA_TAB = Y0 + 2 * T + G + 0.04              # (a) table top
YB = YA_TAB + 3 * RP + 0.14                 # (b, c) top: panel letters, legend of b, title of c
YB_AX, BAH = YB + 0.41, 0.6                 # (b) plot top (below the two legend rows) and height
XB1 = 4.76                                  # (b) right edge of the plot: 27 protein names at an 11.5-pt pitch
XC, CAX = 5.52, 0.92                        # (c) left edge and size of the square plot (top aligned with b's plot)
YD = YB_AX + BAH + 0.83                     # (d) top: below b's rotated protein names (<= 0.70 in) and c's x label
YD_P, RH = YD + 0.31, 0.2                   # (d) plot top, row pitch (as figS1b_stage2_realism.py: title 0.31 in above)
H = round(YD_P + 7 * RH + 0.46, 2)          # (d) seven rows, x tick labels and x label
fig = plt.figure(figsize=(W, H)); REND = fig.canvas.get_renderer()
PAN = {}  # panel id -> axes, for the alignment audit


def txt(x, y, s, **kw):
    """figure text at (x, y) inches from the top-left corner, 10 pt"""
    kw.setdefault('fontsize', FS); kw.setdefault('color', INK)
    return fig.text(x / W, 1 - y / H, s, **kw)


def square(x, yc, c, s=0.085):
    """colour square, left edge x, vertically centred on yc (inches)"""
    fig.patches.append(Rectangle((x / W, 1 - (yc + s / 2) / H), s / W, s / H, transform=fig.transFigure, fc=c, ec='none',
                                 figure=fig))


def bracket(x0, x1, y, s, color):
    """grey rule over a group of columns with its label above"""
    fig.add_artist(Line2D([x0 / W, x1 / W], [1 - y / H] * 2, color='#9A9A9A', lw=0.6))
    txt((x0 + x1) / 2, y - 0.03, s, ha='center', va='bottom', color=color)


def scalebar_in(ax, um):
    """scale bar at the lower right of an image tile, its length written above it, right-aligned inside the tile
    (local variant of imaging.scalebar, whose centred 10-pt label would run past the edge of the small zoom tiles)"""
    h, w = ax.get_images()[0].get_array().shape[:2]; px = um / UM; x1 = w - 0.06 * w; yb = h - 0.07 * h
    ax.plot([x1 - px, x1], [yb, yb], color='white', lw=1.4, solid_capstyle='butt')
    ax.text(x1, yb - 0.035 * h, f'{um:g} µm', color='white', fontsize=FS, ha='right', va='bottom')


def spearman_table(y0, x_names, names, xs, vals, best):
    """one row per protein: colour square + name at x_names (under the measured tile), then one value per method column
    centred at xs (bold: best in the row); the rotated 'Spearman' label sits in the left margin"""
    for i, nm in enumerate(names):
        yc = y0 + RP * (i + 0.5)
        square(x_names, yc, CMY[i]); txt(x_names + 0.125, yc, nm, ha='left', va='center')
        for xc, v, b in zip(xs, vals, best):
            txt(xc, yc, f'{v[i]:.2f}' if np.isfinite(v[i]) else '–', ha='center', va='center',
                fontweight='bold' if b[i] else 'normal')
    txt(XSP, y0 + 1.5 * RP, 'Spearman', rotation=90, rotation_mode='anchor', ha='center', va='center', color=NEUTRAL)


# ---------------- (a) kidney ----------------
K = np.load(f'{DD}/kidney_rcc_allm.npz', allow_pickle=True); prots = [str(p) for p in K['prots']]
SHOW = ['E-Cadherin', 'CD45', 'Ki-67']; J = [prots.index(p) for p in SHOW]
meas = K['measured']; S0 = K['lab'].shape[0]; Wn, Zn = 120, 52
pos = [meas[J[0]] > np.percentile(meas[J[0]], 85), meas[J[1]] > np.percentile(meas[J[1]], 85), meas[J[2]] > np.percentile(meas[J[2]], 97)]


def best_window(n, y0=0, x0=0, size=S0, step=4):
    best, by, bx = -1, 0, 0
    for y in range(y0, y0 + size - n + 1, step):
        for x in range(x0, x0 + size - n + 1, step):
            v = min(pos[0][y:y + n, x:x + n].mean(), pos[1][y:y + n, x:x + n].mean(), 3 * pos[2][y:y + n, x:x + n].mean())
            if v > best: best, by, bx = v, y, x
    return by, bx


oy, ox = best_window(Wn, step=8); zy, zx = best_window(Zn, oy, ox, Wn, step=2); zy, zx = zy - oy, zx - ox
sw = (slice(oy, oy + Wn), slice(ox, ox + Wn)); labw = K['lab'][sw]; cellm = labw > 0
edge = find_boundaries(labw, mode='inner').astype(float)
zs = (slice(zy, zy + Zn), slice(zx, zx + Zn))
cols = [('measured', 'Measured', None), ('mean', 'Stage 1', 'txcellnet'), ('refined', 'Stage 2', 'txcellnet'),
        ('nichetrans', 'NicheTrans', 'nichetrans'), ('dgat', 'DGAT', 'dgat'), ('spamosaic', 'SpaMosaic', 'spamosaic'), ('totalvi', 'totalVI', 'totalvi')]
xa = [X0 + c * (T + G) for c in range(len(cols))]
label(fig, 0.0, 0.02, 'a'); txt(0.17, 0.02, 'Held-out kidney', va='top')
bracket(xa[1], xa[2] + T, 0.23, 'TxCellNet', INK)
bracket(xa[3], xa[6] + T, 0.23, 'Per-cell predictions, painted into their cells', NEUTRAL)
V = {m: [rho('kidney_rcc', m, p) for p in SHOW] for _, _, m in cols if m}
top = [max(V[m][i] for m in V) for i in range(3)]
for c, (k, title, m) in enumerate(cols):
    x = xa[c]; painted = k in ('nichetrans', 'dgat', 'spamosaic', 'totalvi')
    chs = [K[k][j][sw].astype(np.float32) for j in J]
    rgb = composite(chs, cellm, outline=edge, outside=~cellm if painted else None)
    ax = PAN[f'a{c + 1}'] = axes_in(fig, x, Y0, T, T); show(ax, rgb)
    ax.add_patch(Rectangle((zx - 0.5, zy - 0.5), Zn, Zn, fill=False, ec='white', lw=0.6))
    txt(x + T / 2, Y0 - 0.035, title, ha='center', va='bottom')
    if c == 0: scalebar_in(ax, 20)
    ax = PAN[f'a{c + 1}z'] = axes_in(fig, x, Y0 + T + G, T, T)
    show(ax, composite([a[zs] for a in chs], cellm[zs], outline=edge[zs], outside=~cellm[zs] if painted else None), 'bilinear')
    if c == 0: scalebar_in(ax, 10)
spearman_table(YA_TAB, xa[0], ['E-cadherin', 'CD45', 'Ki-67'], [x + T / 2 for x in xa[1:]], [V[m] for _, _, m in cols[1:]],
               [[round(V[m][i], 2) == round(top[i], 2) for i in range(3)] for _, _, m in cols[1:]])

# ---------------- (b) within-cell Pearson per protein ----------------
D = pd.read_csv(f'{DD}/image_metrics_kidney_fin.csv')
M = D.groupby(['method', 'protein'])[['pix_r', 'within_r', 'enrich', 'enrich_measured']].mean().reset_index()
label(fig, 0.0, YB, 'b')
ax = PAN['b'] = axes_in(fig, X0, YB_AX, XB1 - X0, BAH)
order = M[M.method == 'TxCellNet stage 1 (mean)'].sort_values('within_r', ascending=False).protein.tolist(); xx = np.arange(len(order))
s1 = M[M.method == 'TxCellNet stage 1 (mean)'].set_index('protein').reindex(order).within_r.values
s2 = M[M.method == 'TxCellNet stage 2 (refined)'].set_index('protein').reindex(order).within_r.values
cg = M[M.method == 'cognate mRNA (density)'].set_index('protein').reindex(order).within_r.values
ax.vlines(xx, 0, s1, color=HERO, lw=1.0, zorder=2)
ax.scatter(xx, s1, s=11, c=HERO, marker='D', edgecolors='white', linewidths=0.3, zorder=4, label='TxCellNet stage 1')
ax.scatter(xx, s2, s=9, c='white', marker='D', edgecolors=HERO_LIGHT, linewidths=0.8, zorder=3, label='TxCellNet stage 2')
ax.scatter(xx, cg, s=7, c='#B5B5B5', marker='o', edgecolors='none', zorder=3, label='Cognate mRNA density')
ax.axhline(0, color=COL['nichetrans'], lw=0.8, ls=(0, (3, 1.5)), zorder=1, label='Per-cell methods: 0')
ax.set_xticks(xx, [PRETTY.get(p, p) for p in order], rotation=90, rotation_mode='anchor', ha='right', va='center'); ax.tick_params(axis='x', length=0, pad=2)
ax.set_xlim(-0.5, len(order) - 0.5); ax.set_ylim(-0.12, 0.85); ax.set_yticks([0, 0.4, 0.8], ['0', '0.4', '0.8'])
ax.set_ylabel('Within-cell r', labelpad=3)
ax.grid(axis='y', color='#EEEEEE', lw=0.5); ax.set_axisbelow(True)
# two legend rows right of the panel letter (columns: TxCellNet stages | references)
ax.legend(loc='upper left', bbox_to_anchor=(0.17 / W, 1 - (YB - 0.005) / H), bbox_transform=fig.transFigure, ncol=2,
          handlelength=1.2, handletextpad=0.4, columnspacing=1.1, labelspacing=0.3, borderaxespad=0.0, borderpad=0.0,
          markerscale=1.3)

# ---------------- (c) boundary vs interior ----------------
g = M[M.method == 'TxCellNet stage 1 (mean)']
label(fig, XC - 0.70, YB, 'c'); txt(XC - 0.53, YB, 'Boundary − interior (z)', va='top')
ax = PAN['c'] = axes_in(fig, XC, YB_AX, CAX, CAX)
lim = [min(g.enrich_measured.min(), g.enrich.min()) - 0.2, max(g.enrich_measured.max(), g.enrich.max()) + 0.38]  # room for 'CD45'
ax.plot(lim, lim, color='#BDBDBD', lw=0.6, zorder=1)
ax.axhline(0, color=COL['nichetrans'], lw=0.8, ls=(0, (3, 1.5)), zorder=2)
ax.scatter(g.enrich_measured, g.enrich, s=9, c=HERO, edgecolors='white', linewidths=0.3, zorder=3)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Measured', labelpad=2); ax.set_ylabel('TxCellNet', labelpad=2)
TK = ['−1', '', '0', '0.5']  # −0.5 tick unlabelled: at 10 pt its label would touch '−1'
ax.set_xticks([-1, -0.5, 0, 0.5], TK); ax.set_yticks([-1, -0.5, 0, 0.5], TK)
r_ = np.corrcoef(g.enrich_measured, g.enrich)[0, 1]
ax.text(lim[1] - 0.05, -0.45, f'r = {r_:.2f}', ha='right', va='center', color=INK)  # below the diagonal, above the PCNA label
for _, rr in g.iterrows():
    if rr.protein in ('CD45', 'PCNA'):
        ax.annotate(PRETTY.get(rr.protein, rr.protein), (rr.enrich_measured, rr.enrich), xytext=(3, 1) if rr.protein == 'PCNA' else (0, 3),
                    textcoords='offset points', color=INK, va='center' if rr.protein == 'PCNA' else 'bottom', ha='left' if rr.protein == 'PCNA' else 'right')

# ---------------- (d) stage-2 realism: code and numbers of figS1b_stage2_realism.py, placed at YD ----------------
R = pd.read_csv(REAL)
assert set(R.section) == {'rcc', 'ccrcc'}, 'both kidney sections expected'
# (block id, title, rows); blocks b and c share the right-hand column
BLOCKS = [('a', 'Whole image', [('W1', 'Pixel intensity W1'), ('glcm_contrast_d3_fixed', 'GLCM contrast'),
                                ('glcm_homogeneity_d3_fixed', 'GLCM homogeneity'), ('glcm_energy_d3_fixed', 'GLCM energy'),
                                ('glcm_correlation_d3_fixed', 'GLCM correlation'), ('swd', 'Patch SWD'),
                                ('kid_cellpose', 'Cellpose KID')]),
          ('b', 'Per cell', [('cell_sd', 'Intensity SD'), ('cell_cv', 'Intensity CV'), ('cell_edge_centre', 'Edge / center'),
                             ('cell_gs1', 'Fine granularity')]),
          ('c', 'Correlation with measured (1 − r)', [('pix_r', 'Pixel'), ('within_r', 'Within cell')])]
NR = len(BLOCKS[0][2])
assert len(BLOCKS[1][2]) + 1 + len(BLOCKS[2][2]) == NR   # right column: per-cell rows, title row, correlation rows

ratio, nratio, out = {}, {}, []
for _, _, rows in BLOCKS:
    for m, lab in rows:
        gr = R[R.metric == m].groupby('protein')[['dist_stage1', 'dist_stage2', 'dist_noise']].mean()
        assert len(gr) == 27 and gr.notna().all().all(), m
        r = gr.dist_stage2 / gr.dist_stage1; rn = gr.dist_noise / gr.dist_stage1; ratio[m], nratio[m] = r, rn
        out.append(dict(metric=m, label=lab, median_dist_stage1=gr.dist_stage1.median(), median_dist_stage2=gr.dist_stage2.median(),
                        median_dist_noise=gr.dist_noise.median(), median_ratio=r.median(), median_ratio_noise=rn.median(),
                        n_stage2_closer=int((r < 1).sum()), n_stage2_closer_than_noise=int((gr.dist_stage2 < gr.dist_noise).sum()),
                        n=len(r)))
allr = pd.concat(ratio.values()); nmed = pd.Series({m: v.median() for m, v in nratio.items()})
XL = (0.018, 10.0)
assert allr.min() > XL[0] and allr.max() < XL[1], f'ratios outside the axis: {allr.min():.3f}..{allr.max():.3f}'
assert nmed.min() > XL[0] and nmed.max() < XL[1], f'control medians outside the axis: {nmed.min():.3f}..{nmed.max():.3f}'


def width_in(s):
    t = fig.text(0, 0, s, fontsize=FS); w = t.get_window_extent(REND).width / fig.dpi; t.remove(); return w


LABW = [max(width_in(lab) for _, lab in BLOCKS[0][2]), max(width_in(lab) for _, _, rows in BLOCKS[1:] for _, lab in rows)]
CNTW = max(width_in('27/27'), width_in('Closer'))
GL, GC, GCOL = 0.10, 0.08, 0.30       # label-plot gap, plot-count gap, gap between the columns
PW = (W - 0.02 - sum(LABW) - 2 * (GL + GC + CNTW) - GCOL) / 2   # equal plot widths filling the figure width
XA = [LABW[0] + GL, LABW[0] + GL + PW + GC + CNTW + GCOL + LABW[1] + GL]   # plot left edges of the two columns
NB = len(BLOCKS[1][2])
FRAME = {'a': (0, YD_P, NR), 'b': (1, YD_P, NB), 'c': (1, YD_P + (NB + 1) * RH, NR - NB - 1)}   # column, top (in), rows
label(fig, 0.0, YD, 'd')
for bid, title, rows in BLOCKS:
    col, top, nr = FRAME[bid]; xp = XA[col]; x0 = xp - GL - LABW[col]
    ax = PAN[f'd{bid}'] = axes_in(fig, xp, top, PW, nr * RH)
    if top == YD_P:                   # block title: top line of the panel (left block after the panel letter), or centred
        txt(x0 + (0.17 if col == 0 else 0.0), YD, title, va='top', ha='left')      # in the free row above the block
    else:
        txt(x0, top - RH / 2, title, va='center_baseline', ha='left')
    vt = blended_transform_factory(fig.transFigure, ax.transData)
    rng = np.random.default_rng(0)
    for i, (m, lab) in enumerate(rows):
        y = nr - 1 - i; r = ratio[m].values
        ax.scatter(r, y + rng.uniform(-0.25, 0.25, len(r)), s=8, c=HERO_LIGHT, edgecolors='none', zorder=3)
        ax.plot([np.median(r)] * 2, [y - 0.38, y + 0.38], color=HERO, lw=1.8, solid_capstyle='butt', zorder=4)
        ax.plot(nmed[m], y, marker='D', ms=4.6, mfc='white', mec=NEUTRAL, mew=1.0, ls='none', zorder=5)
        ax.text((xp + PW + GC) / W, y, f'{int((r < 1).sum())}/{len(r)}', fontsize=FS, va='center', ha='left', transform=vt, color=INK)
    ax.set_xscale('log'); ax.set_xlim(*XL); ax.set_ylim(-0.5, nr - 0.5)
    ax.set_yticks(range(nr), [lab for _, lab in rows][::-1], fontsize=FS)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.axvline(1, color=INK, lw=0.7, ls=(0, (3, 1.5)), zorder=2)
    if top == YD_P:
        ax.text(1, nr - 0.5, 'Stage 1', fontsize=FS, color=NEUTRAL, ha='center', va='bottom')
        ax.text((xp + PW + GC) / W, nr - 0.5, 'Closer', fontsize=FS, color=NEUTRAL, ha='left', va='bottom', transform=vt)
    if bid == 'b':                    # the correlation block below carries the x axis of the right column
        ax.spines['bottom'].set_visible(False); ax.set_xticks([]); ax.minorticks_off()
    else:
        ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"]); ax.set_xticks([0.03, 0.3, 3], minor=True)
        ax.tick_params(axis='x', which='minor', labelbottom=False)
        ax.set_xlabel('Distance ratio, stage 2 / stage 1', labelpad=2)
# key of the control marker: top line, right-aligned with the left column's count column
kx = XA[0] + PW + GC + CNTW
t = txt(kx, YD, 'Stage 1 + matched noise', va='top', ha='right')
tw = t.get_window_extent(REND).width / fig.dpi
fig.add_artist(Line2D([(kx - tw - 0.09) / W], [1 - (YD + 0.075) / H], marker='D', ms=4.6, mfc='white', mec=NEUTRAL, mew=1.0,
                      ls='none', transform=fig.transFigure))

save(fig, OUT, {'a_kidney_spearman': pd.DataFrame(V, index=SHOW).round(4),
                'b_within_cell': pd.DataFrame({'stage1': s1, 'stage2': s2, 'cognate_mRNA': cg}, index=order).round(4),
                'c_enrichment': g.set_index('protein')[['enrich_measured', 'enrich']].round(4),
                'd_realism_ratios': pd.DataFrame({m: ratio[m] for m in ratio}).round(5),
                'd_realism_ratios_noise': pd.DataFrame({m: nratio[m] for m in nratio}).round(5),
                'd_realism_summary': pd.DataFrame(out).set_index('metric').round(4)},
     panels=PAN, row_groups=[[f'a{i}' for i in range(1, 8)], [f'a{i}z' for i in range(1, 8)]],
     column_groups=[[f'a{i}', f'a{i}z'] for i in range(1, 8)] + [['db', 'dc']])
print('H', H, 'kidney window', oy, ox, 'zoom', zy, zx, 'r', round(r_, 3))
print(pd.DataFrame(V, index=SHOW).round(3).to_string())
print(pd.DataFrame(out).set_index('metric')[['median_ratio', 'median_ratio_noise', 'n_stage2_closer']].round(3).to_string())
