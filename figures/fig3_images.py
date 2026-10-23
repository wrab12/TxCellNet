"""Figure 3: what an image adds over per-cell predictions.
(a) held-out kidney window as a three-protein composite on black (cyan E-cadherin: epithelial membrane; magenta CD45:
    immune-cell membrane; yellow Ki-67: nucleus of proliferating cells): measured | TxCellNet stage 1 | stage 2 | NicheTrans |
    DGAT | SpaMosaic | totalVI (per-cell predictions painted into their cells); second row: zoom (box); under the tiles a
    small table, one row per protein (colour square + name under 'Measured'), one column per method: cell-level Spearman of
    each protein over all held-out cells of the section (bold: best in the row).
(b) colorectal liver metastasis with same-section IMC, kidney-trained models: PanCK, CD45, CD68; same table below.
(c) within-cell Pearson per protein (pixel minus cell mean), held-out kidney; painted per-cell predictions are 0.
(d) boundary-minus-interior enrichment per protein (z), TxCellNet stage 1 against measured, held-out kidney.
Each channel of each panel is scaled between its own median and 99.5th percentile over the panel's cell pixels (display only).
Layout (10 pt): a full width; b (left) and d (right) share the middle band; c full width at the bottom. Every printed glyph
>= 10 pt (figstyle.print_min10), panel letters 11 pt. The protein colour key is the first column of each Spearman table;
the compartment of each kidney protein is given in the caption.
usage: fig3_images.py DATA_DIR OUT_PREFIX
"""
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from skimage.segmentation import find_boundaries
import figstyle
figstyle.print_min10()
from figstyle import W, LS, label, axes_in, save, INK, COL, HERO, HERO_LIGHT, NEUTRAL
from imaging import CMY, composite, show, UM

FS = 10  # every text in this figure (panel letters: figstyle.PANEL_FS = 11)
DD, OUT = sys.argv[1:3]
MET = pd.read_csv(f'{DD}/metrics_allm.csv')
rho = lambda case, m, p: MET[(MET.case == case) & (MET.method == m) & (MET.marker == p)].spearman.mean()
PRETTY = {'E-Cadherin': 'E-cadherin', 'Beta-catenin': 'β-catenin', 'alphaSMA': 'αSMA', 'GranzymeB': 'GzmB', 'CD8A': 'CD8',
          'CD3E': 'CD3'}

# ---------------- geometry (inches from the top-left corner) ----------------
X0, T, G, Y0 = 0.45, 0.78, 0.045, 0.45     # (a) image tiles: left edge, size, gap, top edge
XSP = 0.27                                  # centre of the rotated 'Spearman' label left of each table
RP = 0.16                                   # row pitch of the Spearman tables
TB, GB = 0.68, 0.05                         # (b) tile size and gap (b starts at X0; pitch keeps the 10-pt titles apart)
XD, DAX = 5.42, 1.0                         # (d) left edge and size of the plot area
YA_TAB = Y0 + 2 * T + G + 0.04              # (a) table top
YB = YA_TAB + 3 * RP + 0.14                 # (b, d) top
YB_T = YB + 0.40                            # (b) tile top
YD = YB + 0.24                              # (d) plot top
YC = max(YD + DAX + 0.46, YB_T + TB + 0.04 + 1.5 * RP + 0.32 + 0.07)  # (c) top: below d's x label and b's 'Spearman'
YC_AX, CAH = YC + 0.27, 0.6                 # (c) plot top and height
H = round(YC_AX + CAH + 0.73, 2)              # rotated protein names of (c): <= 0.70 in
fig = plt.figure(figsize=(W, H)); REND = fig.canvas.get_renderer()
PAN = {}  # panel id -> axes, for the optional alignment check


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

# ---------------- (b) CRLM, IMC ----------------
C = np.load(f'{DD}/crlm_tumor3_A3_allm.npz'); marks = [str(mm) for mm in C['marks']]; cellc = C['lab'] > 0
SB = ['PanCK', 'CD45', 'CD68']; JB = [marks.index(p) for p in SB]
dcols = [('measured', 'Measured', None), ('stage1', 'TxCellNet', 'txcellnet'), ('nichetrans', 'NicheTrans', 'nichetrans'),
         ('dgat', 'DGAT', 'dgat'), ('spamosaic', 'SpaMosaic', 'spamosaic'), ('totalvi', 'totalVI', 'totalvi')]
xb = [X0 + c * (TB + GB) for c in range(len(dcols))]
label(fig, 0.0, YB, 'b')
txt(0.17, YB, 'Liver metastasis, IMC; models trained on kidney only', va='top')
VB = {m: [rho('crlm_tumor3_A3', m, p) for p in SB] for _, _, m in dcols if m}
topb = [max(VB[m][i] for m in VB) for i in range(3)]
edgec = find_boundaries(C['lab'], mode='inner').astype(float)
# title nudges (in): 'TxCellNet' and 'NicheTrans' are the widest neighbours, so they move apart (about 10 pt of white
# between them) and 'Measured' follows 'TxCellNet'; every title stays over its own tile
DXB = [-0.045, -0.035, 0.035, 0.01, 0.0, 0.0]
for c, (k, title, m) in enumerate(dcols):
    x = xb[c]; painted = k in ('nichetrans', 'dgat', 'spamosaic', 'totalvi')
    ax = PAN[f'b{c + 1}'] = axes_in(fig, x, YB_T, TB, TB)
    show(ax, composite([C[k][j].astype(np.float32) for j in JB], cellc, outline=edgec * 0.6, outside=~cellc if painted else None))
    txt(x + TB / 2 + DXB[c], YB_T - 0.035, title, ha='center', va='bottom')
    if c == 0: scalebar_in(ax, 50)
spearman_table(YB_T + TB + 0.04, xb[0], SB, [x + TB / 2 for x in xb[1:]], [VB[m] for _, _, m in dcols[1:]],
               [[round(VB[m][i], 2) == round(topb[i], 2) for i in range(3)] for _, _, m in dcols[1:]])

# ---------------- (d) boundary vs interior ----------------
D = pd.read_csv(f'{DD}/image_metrics_kidney_fin.csv')
M = D.groupby(['method', 'protein'])[['pix_r', 'within_r', 'enrich', 'enrich_measured']].mean().reset_index()
g = M[M.method == 'TxCellNet stage 1 (mean)']
label(fig, XD - 0.70, YB, 'd'); txt(XD - 0.53, YB, 'Boundary − interior (z)', va='top')
ax = PAN['d'] = axes_in(fig, XD, YD, DAX, DAX)
lim = [min(g.enrich_measured.min(), g.enrich.min()) - 0.2, max(g.enrich_measured.max(), g.enrich.max()) + 0.32]
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

# ---------------- (c) within-cell Pearson per protein ----------------
label(fig, 0.0, YC, 'c')
ax = PAN['c'] = axes_in(fig, X0, YC_AX, W - 0.05 - X0, CAH)
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
ax.set_xlim(-0.6, len(order) - 0.4); ax.set_ylim(-0.12, 0.85); ax.set_yticks([0, 0.4, 0.8], ['0', '0.4', '0.8'])
ax.set_ylabel('Within-cell r', labelpad=3)
ax.grid(axis='y', color='#EEEEEE', lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='upper left', bbox_to_anchor=(0.17 / W, 1 - (YC - 0.005) / H), bbox_transform=fig.transFigure, ncol=4,
          handlelength=1.2, handletextpad=0.4, columnspacing=1.1, borderaxespad=0.0, borderpad=0.0, markerscale=1.3)
save(fig, OUT, {'a_kidney_spearman': pd.DataFrame(V, index=SHOW).round(4), 'b_crlm_spearman': pd.DataFrame(VB, index=SB).round(4),
                'c_within_cell': pd.DataFrame({'stage1': s1, 'stage2': s2, 'cognate_mRNA': cg}, index=order).round(4),
                'd_enrichment': g.set_index('protein')[['enrich_measured', 'enrich']].round(4)},
     panels=PAN, row_groups=[[f'a{i}' for i in range(1, 8)], [f'a{i}z' for i in range(1, 8)], [f'b{i}' for i in range(1, 7)]],
     column_groups=[[f'a{i}', f'a{i}z'] for i in range(1, 8)])
print('H', H, 'kidney window', oy, ox, 'zoom', zy, zx, 'r', round(r_, 3))
print(pd.DataFrame(V, index=SHOW).round(3).to_string()); print(pd.DataFrame(VB, index=SB).round(3).to_string())
