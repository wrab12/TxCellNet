"""Supplementary Fig. S3: predicted images of every compared method, in the windows of Fig. 3a
(held-out kidney) and Fig. 3b (colorectal liver metastasis with same-section IMC, kidney-trained models). Three-protein
composites as in Fig. 3 (each channel scaled from its median to its 99.5th percentile over the panel's cell pixels); columns:
measured, TxCellNet stage 1 and stage 2, the published methods and the baselines (per-cell predictions painted into their
cells), and the cognate mRNA (Gaussian-smoothed density of the molecules of the coding genes, sigma = 2 pixels). Under each
row: cell-level Spearman of each protein over all scored cells of the section (bold: best).
Layout (10 pt): twelve columns; the column titles do not fit over the 0.49-in tiles at 10 pt,
so they are set vertically (bottom to top, at most two lines); the protein colour key sits in the first column of
each table of numbers (under 'Measured', which has no value), as in Fig. 3. 'Painted into cells' is stated in the caption.
Every printed glyph >= 10 pt (figstyle.print_min10), panel letters 11 pt.
usage: figS3_allmethods.py DATA_DIR OUT_PREFIX
"""
import json, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries
import figstyle
figstyle.print_min10()
from figstyle import W, LS, label, axes_in, save, INK, NEUTRAL
from imaging import CMY, composite, show, scalebar

FS = 10  # every text in this figure (panel letters: figstyle.PANEL_FS = 11)
DD, OUT = sys.argv[1:3]
MET = pd.concat([pd.read_csv(f'{DD}/metrics_allm.csv'), pd.read_csv(f'{DD}/metrics_cognate.csv')])
rho = lambda case, m, p: MET[(MET.case == case) & (MET.method == m) & (MET.marker == p)].spearman.mean()
PRETTY = {'E-Cadherin': 'E-cadherin'}
COLS = [('measured', 'Measured', None), ('s1', 'TxCellNet\nstage 1', 'txcellnet'), ('s2', 'TxCellNet\nstage 2', 'txcellnet'),
        ('nichetrans', 'NicheTrans', 'nichetrans'), ('dgat', 'DGAT', 'dgat'), ('spamosaic', 'SpaMosaic', 'spamosaic'),
        ('totalvi', 'totalVI', 'totalvi'), ('mlp', 'MLP', 'mlp'), ('ridge', 'Ridge', 'ridge'), ('knn', 'Reference\nmapping', 'knn'),
        ('celltype', 'Cell-type\nlookup', 'celltype'), ('cognate', 'Cognate\nmRNA', 'cognate')]
PAINTED = {'nichetrans', 'dgat', 'spamosaic', 'totalvi', 'mlp', 'ridge', 'knn', 'celltype'}

# ---------------- geometry (inches from the top-left corner) ----------------
# X0: left edge of the 'Measured' tiles; the protein names of the tables (from XK) run under that column and end before the
# numbers of the stage-1 column start
X0, G = 0.33, 0.03; T = (W - X0 - 11 * G) / 12   # tile size (0.487 in)
XK = 0.03                                          # colour square of the protein names
RP = 0.16                                          # row pitch of the tables of numbers
HB = 0.75                                          # band of the vertical column titles (longest: 'NicheTrans', 0.71 in)
GT, GR = 0.04, 0.09                                # tile -> its numbers; numbers -> next tile row
YA_RULE = 0.43                                     # (a) rule of the 'Published methods' / 'Baselines' brackets
YA_T = YA_RULE + 0.05 + HB + 0.04                  # (a) top of the first tile row
xcol = lambda c: X0 + c * (T + G)


def row_h(n):  # one tile row with n rows of numbers under it
    return T + GT + n * RP


YA_T2 = YA_T + row_h(3) + GR
YB = YA_T2 + row_h(3) + 0.13                       # (b) letter and title
YB_T = YB + 0.25 + HB + 0.04                       # (b) top of the first tile row
YB_T2 = YB_T + row_h(3) + GR
H = round(YB_T2 + row_h(2) + 0.03, 2)
fig = plt.figure(figsize=(W, H)); REND = fig.canvas.get_renderer()
PAN = {}  # panel id -> axes, for the optional alignment check


def txt(x, y, s, **kw):
    """figure text at (x, y) inches from the top-left corner, 10 pt"""
    kw.setdefault('fontsize', FS); kw.setdefault('color', INK)
    return fig.text(x / W, 1 - y / H, s, **kw)


def headers(y, meas_title):
    """vertical column titles (bottom to top), bottom edge at y, centred over each column"""
    for c, (k, t, m) in enumerate(COLS):
        # anchor mode: the start of the title (left in its own frame) sits on y, the block of lines is centred on the column
        txt(xcol(c) + T / 2, y, meas_title if k == 'measured' else t, rotation=90, rotation_mode='anchor', ha='left',
            va='center', multialignment='left', linespacing=LS)


def block_row(y, case, prots, get, cellm, edge, bar_um=None, rid='r'):
    """one row of composites at y (inches from the top) and the rows of numbers under it; returns the y below"""
    for c, (k, t, m) in enumerate(COLS):
        chs = [get(k, p) if p else np.zeros_like(cellm, np.float32) for p in prots]
        ax = PAN[f'{rid}{c + 1}'] = axes_in(fig, xcol(c), y, T, T)
        show(ax, composite(chs, cellm, outline=edge, outside=~cellm if k in PAINTED else None))
        if c == 0 and bar_um: scalebar(ax, bar_um, text=False)
    yy = y + T + GT
    for i, p in enumerate(prots):
        if not p: continue
        vals = {m: rho(case, m, p) for _, _, m in COLS if m}; top = np.nanmax(list(vals.values()))
        yc = yy + RP / 2; s = 0.085
        fig.patches.append(Rectangle((XK / W, 1 - (yc + s / 2) / H), s / W, s / H, transform=fig.transFigure, fc=CMY[i],
                                     ec='none', figure=fig))
        txt(XK + 0.125, yc, PRETTY.get(p, p), va='center', ha='left')
        for c, (k, t, m) in enumerate(COLS):
            if m is None: continue
            v = vals[m]
            txt(xcol(c) + T / 2, yc, '–' if not np.isfinite(v) else f'{v:.2f}'.replace('-0.00', '0.00'), ha='center',
                va='center', fontweight='bold' if np.isfinite(v) and round(v, 2) == round(top, 2) else 'normal')
        yy += RP
    return yy


# ---------------- (a) held-out kidney, window of Fig. 3a ----------------
K = np.load(f'{DD}/kidney_rcc_allm.npz', allow_pickle=True); prots = [str(p) for p in K['prots']]; genes = [str(g) for g in K['genes']]
own = json.loads(str(K['own'][0])); S0 = K['lab'].shape[0]; meas = K['measured']; Wn = 120
J = [prots.index(p) for p in ['E-Cadherin', 'CD45', 'Ki-67']]
pos = [meas[J[0]] > np.percentile(meas[J[0]], 85), meas[J[1]] > np.percentile(meas[J[1]], 85), meas[J[2]] > np.percentile(meas[J[2]], 97)]
best, oy, ox = -1, 0, 0
for y in range(0, S0 - Wn + 1, 8):  # same selection as fig3_images.py
    for x in range(0, S0 - Wn + 1, 8):
        v = min(pos[0][y:y + Wn, x:x + Wn].mean(), pos[1][y:y + Wn, x:x + Wn].mean(), 3 * pos[2][y:y + Wn, x:x + Wn].mean())
        if v > best: best, oy, ox = v, y, x
sw = (slice(oy, oy + Wn), slice(ox, ox + Wn)); labw = K['lab'][sw]; cellk = labw > 0; edgek = find_boundaries(labw, mode='inner').astype(float)
ty, tx, tg = K['ty'], K['tx'], K['tg']


def kid(k, p):
    j = prots.index(p)
    if k == 'cognate':
        gs = [genes.index(g) for g in own.get(p, []) if g in genes]; m = np.isin(tg, gs)
        d = np.zeros((S0, S0), np.float32); np.add.at(d, (ty[m], tx[m]), 1)
        return gaussian_filter(d, 2.0)[sw]
    return K[{'s1': 'mean', 's2': 'refined'}.get(k, k)][j][sw].astype(np.float32)


label(fig, 0.0, 0.02, 'a')
txt(0.17, 0.02, f'Held-out kidney, window of Fig. 3a ({Wn * 0.85:.0f} µm)', va='top')
for a_, b_, lab in [(3, 6, 'Published methods'), (7, 10, 'Baselines')]:
    xa, xb = xcol(a_), xcol(b_) + T
    fig.add_artist(Line2D([xa / W, xb / W], [1 - YA_RULE / H] * 2, color='#9A9A9A', lw=0.6))
    txt((xa + xb) / 2, YA_RULE - 0.03, lab, ha='center', va='bottom', color=NEUTRAL)
headers(YA_T - 0.04, 'Measured')
y = block_row(YA_T, 'kidney_rcc', ['E-Cadherin', 'CD45', 'Ki-67'], kid, cellk, edgek, bar_um=20, rid='k1_')
assert abs(y + GR - YA_T2) < 1e-9
y = block_row(YA_T2, 'kidney_rcc', ['Vimentin', 'CD68', 'PanCK'], kid, cellk, edgek, rid='k2_')

# ---------------- (b) CRLM, IMC, window of Fig. 3b ----------------
C = np.load(f'{DD}/crlm_tumor3_A3_allm.npz'); marks = [str(m) for m in C['marks']]
CG = np.load(f'{DD}/crlm_tumor3_A3_cognate.npz'); assert [str(m) for m in CG['marks']] == marks
cellc = C['lab'] > 0; edgec = find_boundaries(C['lab'], mode='inner').astype(float) * 0.6


def crlm(k, p):
    i = marks.index(p)
    if k == 'cognate': return CG['cognate'][i]
    return C[{'s1': 'stage1', 's2': 'stage2'}.get(k, k)][i].astype(np.float32)


label(fig, 0.0, YB, 'b')
txt(0.17, YB, 'Liver metastasis, same-section IMC, window of Fig. 3b (255 µm); models trained on kidney', va='top')
headers(YB_T - 0.04, 'Measured\n(IMC)')
y = block_row(YB_T, 'crlm_tumor3_A3', ['PanCK', 'CD45', 'CD68'], crlm, cellc, edgec, bar_um=50, rid='c1_')
assert abs(y + GR - YB_T2) < 1e-9
y = block_row(YB_T2, 'crlm_tumor3_A3', ['Vimentin', None, 'Ki-67'], crlm, cellc, edgec, rid='c2_')
print('height used', round(y, 2), 'of', H, '| tile', round(T, 3), '| kidney window', oy, ox)

# measured extents of the texts (in), to check the title band and the right edge
ext = lambda t: t.get_window_extent(REND).transformed(fig.dpi_scale_trans.inverted())
vt = [t for t in fig.texts if t.get_rotation() == 90]
print('tallest column title', round(max(ext(t).height for t in vt), 3), 'of', HB,
      '| rightmost text edge', round(max(ext(t).x1 for t in fig.texts), 3), 'of', W)
TAB = pd.DataFrame({(case, p): {m: rho(case, m, p) for _, _, m in COLS if m} for case, ps in
                    [('kidney_rcc', ['E-Cadherin', 'CD45', 'Ki-67', 'Vimentin', 'CD68', 'PanCK']),
                     ('crlm_tumor3_A3', ['PanCK', 'CD45', 'CD68', 'Vimentin', 'Ki-67'])] for p in ps}).T
R_ = ['k1_', 'k2_', 'c1_', 'c2_']
save(fig, OUT, {'spearman': TAB.round(4)}, panels=PAN, row_groups=[[f'{r}{c}' for c in range(1, 13)] for r in R_],
     # columns: within each block (a: kidney rows k1, k2; b: CRLM rows c1, c2) and across the blocks (edges only; the gap
     # between the blocks holds the title of b, so the four rows of a column do not have one repeated gutter)
     column_groups=[[f'k1_{c}', f'k2_{c}'] for c in range(1, 13)] + [[f'c1_{c}', f'c2_{c}'] for c in range(1, 13)]
                   + [[f'k1_{c}', f'c1_{c}'] for c in range(1, 13)])
