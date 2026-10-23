"""Benchmark summary table (scIB-style): rows = methods (TxCellNet, published methods, baselines; one fixed order in every
table), columns = groups of sections; cell = mean Spearman, shaded by its rank within the column (darker = better), best per
column in bold; last column = horizontal bar of the overall mean in the method's colour.
The table is laid out in printed inches. The axes span the full figure width and the height of the
table; its data units are inches (x from the left edge of the figure, y down from the top of the table), so every size below
is a printed size. Every text item is drawn at `fs` (10 pt); the panel letter and title sit in the free header cell above
the method names (drawn by the caller). Used only by figS5_tables.py.
"""
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from figstyle import COL, LAB, MK, PUBLISHED, BASELINES, INK, LS

SHADE = mcolors.LinearSegmentedColormap.from_list('shade', ['#F5F6F8', '#C9D3E2', '#5E7799', '#22385C'])

FIXED = [['txcellnet'], ['nichetrans', 'dgat', 'totalvi', 'spamosaic'], ['mlp', 'ridge_niche', 'knn', 'ridge', 'celltype', 'own_mRNA']]

# horizontal layout (inches from the left edge of the 6.5 in figure)
MK_X, NAME_X = 0.07, 0.16          # method marker centre, method name
CELLS = (1.44, 5.60)               # group columns (the longest name, "Reference mapping", ends at 1.38 in)
BAR_X, BAR_L = 5.70, 0.47          # overall-mean bar: start, length (in) per unit of Spearman
PAD_HDR, PAD_GRP, PAD_RULE = 0.035, 0.035, 0.025   # column headers -> cells, group rule -> headers, group label -> rule
CELL_FILL, BAR_FILL, INSET = 0.88, 0.60, 0.03      # cell / bar height (fraction of the row pitch), cell side inset (in)
DY = 0.5 / 72      # row texts 0.5 pt below the row centre: centres the digits (and the font box) in the cell


def order_rows(T, overall='All', fixed=True):
    """one fixed method order in every table (same-section ranking), so that panels can be read against each other"""
    if fixed: return [[m for m in b if m in T.index] for b in FIXED]
    pub = sorted([m for m in PUBLISHED if m in T.index], key=lambda m: -T.loc[m, overall])
    base = sorted([m for m in BASELINES if m in T.index], key=lambda m: -T.loc[m, overall])
    return ['txcellnet'], pub, base


def text_in(fig, s, fs, weight='normal'):
    """rendered (width, height) in inches of a label, multi-line labels at the common leading LS"""
    t = fig.text(0, 0, s, fontsize=fs, linespacing=LS, fontweight=weight); bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def header_height(fig, heads, overall_head, groups=None, fs=10, top=0.01):
    """height (in) of the header block: optional group labels with rules, then the bottom-aligned column headers"""
    ch = max(text_in(fig, s, fs)[1] for s in list(heads) + [overall_head])
    gh = (text_in(fig, groups[0][0], fs)[1] + PAD_RULE + PAD_GRP) if groups else 0.0
    return top + gh + ch + PAD_HDR


def table_height(fig, T, heads, overall_head, groups=None, row_h=0.17, gap=0.3, fs=10):
    blocks = [b for b in order_rows(T) if b]
    return header_height(fig, heads, overall_head, groups, fs) + (sum(map(len, blocks)) + gap * (len(blocks) - 1)) * row_h


def bench_table(ax, T, cols, heads, overall='All', overall_head='All', groups=None, row_h=0.17, gap=0.3, fs=10,
                fmt=lambda v: f'{v:.2f}'):
    """T: DataFrame (method keys x columns incl. `overall`); cols: group columns shown as cells; heads: header strings;
    groups: optional [(title, n_columns), ...] drawn as spanning headers with a rule under each. The axes must span the
    full figure width and table_height(...) inches. Returns [(method, row centre y)] and the y of the header bottom."""
    fig = ax.figure; Wf, Hf = fig.get_size_inches(); pos = ax.get_position()
    w_in, h_in = pos.width * Wf, pos.height * Hf
    ax.set_xlim(0, w_in); ax.set_ylim(h_in, 0); ax.axis('off')
    hdr = header_height(fig, heads, overall_head, groups, fs)
    rows, y = [], hdr
    for block in [b for b in order_rows(T, overall) if b]:
        for m in block:
            rows.append((m, y + row_h / 2)); y += row_h
        y += gap * row_h
    cw = (CELLS[1] - CELLS[0]) / len(cols)
    xs = [CELLS[0] + j * cw for j in range(len(cols))]
    yb = hdr - PAD_HDR                                    # bottom of the column headers
    for h1, h2 in zip(heads[:-1], heads[1:]):            # bottom-aligned header lines: >= 4 pt to the neighbouring header
        for l1, l2 in zip(h1.split('\n')[::-1], h2.split('\n')[::-1]):
            sep = cw - (text_in(fig, l1, fs)[0] + text_in(fig, l2, fs)[0]) / 2
            assert sep * 72 >= 4, f'header lines {l1!r} and {l2!r} only {sep * 72:.1f} pt apart'
    if groups:
        ch = max(text_in(fig, s, fs)[1] for s in list(heads) + [overall_head])
        yr = yb - ch - PAD_GRP; j0 = 0
        for gl, n in groups:
            xa, xb = xs[j0] + INSET + 0.02, xs[j0 + n - 1] + cw - INSET - 0.02
            ax.plot([xa, xb], [yr, yr], color='#9A9A9A', lw=0.6, solid_capstyle='butt')
            ax.text((xa + xb) / 2, yr - PAD_RULE, gl, ha='center', va='bottom', fontsize=fs, color=INK)
            j0 += n
    for j, (c, h) in enumerate(zip(cols, heads)):
        v = T[c].astype(float); fin = v.dropna()
        rk = fin.rank(ascending=True, method='min'); rk = (rk - 1) / max(len(fin) - 1, 1)  # 0 = worst, 1 = best
        best = fin.max(); x = xs[j]
        ax.text(x + cw / 2, yb, h, ha='center', va='bottom', fontsize=fs, color=INK, linespacing=LS)
        for m, yy in rows:
            val = v.get(m, np.nan)
            box = Rectangle((x + INSET, yy - CELL_FILL * row_h / 2), cw - 2 * INSET, CELL_FILL * row_h, ec='none')
            if not np.isfinite(val):
                box.set_facecolor('#FAFAFA'); ax.add_patch(box)
                ax.text(x + cw / 2, yy + DY, '–', ha='center', va='center', fontsize=fs, color='#9A9A9A'); continue
            colr = SHADE(0.08 + 0.92 * rk[m]); lum = np.dot(mcolors.to_rgb(colr), [0.299, 0.587, 0.114])
            box.set_facecolor(colr); ax.add_patch(box)
            ax.text(x + cw / 2, yy + DY, fmt(val), ha='center', va='center', fontsize=fs,
                    color='white' if lum < 0.55 else INK, fontweight='bold' if fmt(val) == fmt(best) else 'normal')
    ax.text(BAR_X, yb, overall_head, ha='left', va='bottom', fontsize=fs, color=INK, linespacing=LS)
    best_all = T[overall].max()
    for m, yy in rows:
        val = T.loc[m, overall]
        if not np.isfinite(val): continue
        bl = BAR_L * max(val, 0)
        ax.add_patch(Rectangle((BAR_X, yy - BAR_FILL * row_h / 2), bl, BAR_FILL * row_h, fc=COL[m], ec='none'))
        ax.text(BAR_X + bl + 0.04, yy + DY, f'{val:.3f}', ha='left', va='center', fontsize=fs, color=INK,
                fontweight='bold' if f'{val:.3f}' == f'{best_all:.3f}' else 'normal')
    for m, yy in rows:
        ax.plot([MK_X], [yy], marker=MK[m], ms=5.2 if m == 'txcellnet' else 4.8, mfc=COL[m],
                mec='white' if m != 'own_mRNA' else '#9A9A9A', mew=0.4, clip_on=False)
        ax.text(NAME_X, yy + DY, LAB[m], ha='left', va='center', fontsize=fs, color=INK, fontweight='bold' if m == 'txcellnet' else 'normal')
    return rows, yb
