"""Shared style for the TxCellNet data figures (v4: v3 audited with the nature-figure 2.8.0 QA contract).

Final size: drawn at the RECOMB text width (6.5 in) and included at 100 %, so the font sizes below are the printed sizes
(Nature guide: sans-serif 5-7 pt, bold lowercase 8 pt panel letters, no outlined text, TrueType-embedded fonts; every
rendered glyph >= 5 pt, so no mathtext superscripts).
Colour: one hero colour for TxCellNet (vermillion), one blue family for the four published methods (dark = strongest),
one grey family for the baselines (Okabe-Ito based, colour-blind safe; red/green pairs avoided); marker shapes double-encode.
Exports: SVG with live text (svg.fonttype none) and PDF with TrueType (Type 42) fonts are the editable masters; PNG is a
preview only. Every figure also writes the plotted numbers as CSV (source data).
QA (v4): multi-line labels use one leading LS (lines must not overlap in the PDF); before export, save() measures the final
panel rectangles and blocks on misalignment (audit_panel_alignment.py of the nature-figure skill, 1.5 pt tolerance) when
that module is importable; the alignment report is written next to the figure (QA only).
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 6.5, 'axes.titlesize': 7, 'axes.labelsize': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5, 'axes.labelpad': 2.0, 'axes.titlepad': 3.0,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'legend.handletextpad': 0.4,
    'lines.linewidth': 1.0, 'lines.markersize': 3.5, 'figure.dpi': 150, 'savefig.dpi': 600,
})
W = 6.5  # figure width (in) = RECOMB text width
PANEL_FS, KEY_FS, SMALL_FS = 8, 6.5, 5.2  # panel letters, inline colour keys, scale bars / value strips


def print_min10():
    """RECOMB 2027 floor: every printed glyph >= 10 pt (main-text figures 2-5). Call right after importing figstyle;
    the supplementary figures keep the 5-7 pt style above."""
    global PANEL_FS, KEY_FS, SMALL_FS
    mpl.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'legend.fontsize': 10, 'lines.markersize': 5, 'xtick.major.size': 3,
                         'ytick.major.size': 3, 'xtick.major.pad': 2, 'ytick.major.pad': 2, 'axes.labelpad': 3})
    PANEL_FS, KEY_FS, SMALL_FS = 11, 10, 10
LS = 1.2  # leading of multi-line labels (x font size): at 0.95 the lines of a two-line label overlapped in the PDF

HERO = '#D55E00'          # TxCellNet (Okabe-Ito vermillion)
HERO_LIGHT = '#F2A578'    # TxCellNet variants (stage 2, clean-only, ...)
# key, label, colour, marker
METHODS = [('txcellnet', 'TxCellNet', HERO, 'D'),
           ('nichetrans', 'NicheTrans', '#1B3A6B', 'o'), ('dgat', 'DGAT', '#3C6BAA', 's'),
           ('spamosaic', 'SpaMosaic', '#79A2D2', '^'), ('totalvi', 'totalVI', '#AFC6E6', 'v'),
           ('mlp', 'MLP', '#4D4D4D', 'o'), ('ridge_niche', 'Ridge + neighborhood', '#4D4D4D', 'o'), ('ridge', 'Ridge', '#7A7A7A', 's'), ('knn', 'Reference mapping', '#A3A3A3', '^'),
           ('celltype', 'Cell-type lookup', '#C6C6C6', 'v'), ('own_mRNA', 'Cognate mRNA', '#DEDEDE', 'o')]
COL = {k: c for k, _, c, _ in METHODS}; LAB = {k: l for k, l, _, _ in METHODS}; MK = {k: m for k, _, _, m in METHODS}
PUBLISHED = ['nichetrans', 'dgat', 'spamosaic', 'totalvi']; BASELINES = ['mlp', 'ridge_niche', 'ridge', 'knn', 'celltype', 'own_mRNA']
NEUTRAL = '#6E6E6E'; LIGHT = '#E6E6E6'; INK = '#222222'


def label(fig, x, y, letter):
    """bold lowercase panel letter at figure coordinates (x, y) in inches from the top-left corner"""
    w, h = fig.get_size_inches()
    fig.text(x / w, 1 - y / h, letter, fontsize=PANEL_FS, fontweight='bold', va='top', ha='left', color=INK)


def axes_in(fig, x, y, w, h):
    """axes at (x, y) inches from the top-left corner, size (w, h) inches"""
    W_, H_ = fig.get_size_inches()
    return fig.add_axes([x / W_, 1 - (y + h) / H_, w / W_, h / H_])


def save(fig, out, data=None, panels=None, row_groups=None, column_groups=None, exemptions=None, qa_dir=None):
    """editable masters (SVG, PDF with TrueType fonts) + PNG preview; data: dict name -> DataFrame (source data CSVs).
    panels: dict panel id -> axes of the comparable plot areas (image tiles, plots; not tables of text, colour keys or
    value strips); row_groups / column_groups: lists of panel ids that must share edges and sizes. With panels given,
    the final layout is measured and a misalignment above 1.5 pt raises before anything is written."""
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    if panels and len(panels) > 1:
        try:
            from audit_panel_alignment import require_matplotlib_panel_alignment
        except ImportError:
            print('alignment audit skipped: audit_panel_alignment.py (nature-figure skill) not importable')
        else:
            qa = qa_dir or os.path.join(os.path.dirname(out) or '.', 'qa'); os.makedirs(qa, exist_ok=True)
            base = os.path.join(qa, os.path.basename(out))
            res = require_matplotlib_panel_alignment(fig, axes=list(panels.values()), panel_ids=list(panels.keys()),
                                                     row_groups=row_groups or [], column_groups=column_groups or [],
                                                     exemptions=exemptions or [], json_out=base + '.alignment.json',
                                                     overlay_svg=base + '.alignment.svg', tolerance_pt=1.5,
                                                     gutter_tolerance_pt=1.5, strict=True)
            print('panel alignment:', res.get('verdict', res.get('status', '?')))
    fig.savefig(out + '.svg'); fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=300)
    for name, df in (data or {}).items(): df.to_csv(f'{out}_data_{name}.csv')
    plt.close(fig)


def key(fig, x, y, items, fs=None, sq=0.075, pad=0.05, sep=0.16):
    """inline colour key at (x, y) inches from the top-left: items = [(colour, text), ...]; returns the x where it ends"""
    w, h = fig.get_size_inches(); r = fig.canvas.get_renderer(); fs = KEY_FS if fs is None else fs
    from matplotlib.patches import Rectangle
    for c, s in items:
        fig.patches.append(Rectangle((x / w, 1 - (y + sq / 2 + 0.005) / h), sq / w, sq / h, transform=fig.transFigure, fc=c, ec='none', figure=fig))
        t = fig.text((x + sq + pad) / w, 1 - y / h, s, fontsize=fs, va='center', ha='left', color=INK)
        x += sq + pad + t.get_window_extent(r).width / fig.dpi + sep
    return x
