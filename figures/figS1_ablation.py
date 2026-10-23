"""Supplementary Fig. S1: what each component contributes, and what stage 2 adds.
(a) ablations on the 32 external Xenium sections: change of the median Spearman per section relative to one network trained
    with half of its patches randomized (mean of its three seeds); dots: sections, bar: mean, diamond: kidney held-out stripes
(b) stage 2 vs stage 1 on the kidney held-out stripes: median over proteins and both sections of image statistics relative to
    the measured image (1 = measured) and correlations
(c) radially averaged power spectra (mean over proteins) of a held-out window
Layout (10 pt): a spans the full width on top (row names left, plot, mean and sections better right); b and c share the
bottom row. Every printed glyph >= 10 pt (figstyle.print_min10); panel letters 11 pt.
usage: figS1_ablation.py PER_DATASET_CSV REFINE_METRICS_CSV CROPS_NPZ OUT_PREFIX
"""
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import figstyle
figstyle.print_min10()
from figstyle import W, LS, label, axes_in, save, INK, HERO, HERO_LIGHT, NEUTRAL

FS = 10  # every text in this figure (panel letters: figstyle.PANEL_FS = 11)
TCSV, RCSV, NPZ, OUT = sys.argv[1:5]
T = pd.read_csv(TCSV, index_col=0)
KID = ['kidney_rcc', 'kidney_ccrcc']
SEEDS = ['pm_full_s1', 'pm_full_s2', 'pm_full_s3']  # one mixed-regime network, three seeds: the reference is their mean
T['ref'] = T[SEEDS].mean(1)
ROWS = [('Same network, single seeds', SEEDS, 'each'), ('Retrained per panel (2 seeds)', ['pm_s1', 'pm_s2'], 'avgpair'),
        ('Clean regime only (3 seeds)', ['fin_clean_s1', 'fin_clean_s2', 'fin_clean_s3'], 'avg'),
        ('Randomized regime only (3 seeds)', ['fin_rand_s1', 'fin_rand_s2', 'fin_rand_s3'], 'avg'),
        ('No DAPI', ['abl_nodapi_full'], 'avg'), ('No niche transformer', ['abl_noniche_full'], 'avg'),
        ('Fixed PCA gene vectors', ['abl_pca_full'], 'avg'), ('Ensemble: 3 mixed networks', ['tx_mix3'], 'avg'),
        ('Ensemble: 1 clean + 1 randomized', ['tx_c1r1'], 'avg'), ('TxCellNet: 3 clean + 3 randomized', ['txcellnet'], 'avg')]
EXT = [c for c in T.index if c not in KID and not c.startswith('vhd') and T.loc[c, SEEDS].notna().all()]


def delta(members, kind, cases):
    if not all(m in T for m in members): return pd.Series(dtype=float)
    if kind == 'each': return pd.concat([T.loc[cases, m] - T.loc[cases, 'ref'] for m in members]).dropna()
    if kind == 'avgpair':  # panel-matched seeds vs the same seeds of the unrestricted network
        d = (T.loc[cases, members].mean(1) - T.loc[cases, ['pm_full_s1', 'pm_full_s2']].mean(1)).dropna()
        return d.drop([c for c in d.index if c.startswith('vhd')], errors='ignore')
    return (T.loc[cases, members].mean(1) - T.loc[cases, 'ref']).dropna()


def signed(v):  # typographic minus for the printed values (the source data keep the plain numbers)
    return f'{v:+.3f}'.replace('-', '−')


H = 6.05
fig = plt.figure(figsize=(W, H))
rend = fig.canvas.get_renderer()


def width_in(s, **kw):
    """printed width (in) of a 10 pt string"""
    t = fig.text(0, 0, s, fontsize=FS, **kw); w = t.get_window_extent(rend).width / fig.dpi; t.remove(); return w


# ---- a: ablations, full width ----
DA = {}                                 # row -> (sections, kidney stripes, printed value)
for lab, members, kind in ROWS:
    d = delta(members, kind, EXT); dk = delta(members, kind, KID)
    if len(d): DA[lab] = (d, dk, signed(d.mean()) if kind == 'each' else f'{signed(d.mean())}  ({(d > 0).sum()}/{len(d)})')
NR = len(ROWS)
HEAD = 'Mean (sections better)'
LAB_W = max(width_in(r[0], fontweight='bold' if r[0].startswith('TxCellNet') else 'normal') for r in ROWS)
VAL_W = max(width_in(HEAD), max(width_in(t, fontweight='bold') for *_, t in DA.values()))
YA, RH = 0.48, 0.235                    # plot top, row height (in)
XA = 0.02 + LAB_W + 0.10                # plot left: row names right-aligned against it
XV = W - 0.05 - VAL_W                   # value column (left-aligned)
WA = XV - 0.14 - XA
label(fig, 0.0, 0.02, 'a'); fig.text(0.17 / W, 1 - 0.03 / H, f'Ablations on the {len(EXT)} external Xenium sections', fontsize=FS,
                                     va='top', color=INK)
ax = axA = axes_in(fig, XA, YA, WA, NR * RH)
vt = blended_transform_factory(fig.transFigure, ax.transData)
rng = np.random.default_rng(0); out = {}
for i, (lab, members, kind) in enumerate(ROWS):
    if lab not in DA: continue
    y = NR - 1 - i; d, dk, txt = DA[lab]; hero = lab.startswith('TxCellNet')
    ax.scatter(d.values, y + rng.uniform(-0.22, 0.22, len(d)), s=9, c=HERO_LIGHT if hero else '#B5B5B5', edgecolors='none', zorder=3)
    ax.plot([d.mean()] * 2, [y - 0.36, y + 0.36], color=HERO if hero else INK, lw=1.8, zorder=4, solid_capstyle='butt')
    if len(dk): ax.scatter(dk.mean(), y, s=24, marker='D', c='white', edgecolors=HERO if hero else INK, linewidths=0.9, zorder=5)
    ax.text(XV / W, y, txt, fontsize=FS, va='center', ha='left', transform=vt, color=INK, fontweight='bold' if hero else 'normal')
    out[lab] = dict(mean=d.mean(), better=int((d > 0).sum()), n=len(d), kidney=dk.mean() if len(dk) else np.nan)
ax.axvline(0, color='#9A9A9A', lw=0.6, zorder=1)
ax.set_yticks(range(NR), [r[0] for r in ROWS][::-1], fontsize=FS)
for t in ax.get_yticklabels():
    if t.get_text().startswith('TxCellNet'): t.set_fontweight('bold')
ax.set_xlim(-0.315, 0.315); ax.set_xticks([-0.3, -0.15, 0, 0.15, 0.3]); ax.set_ylim(-0.6, NR - 0.4)
ax.tick_params(axis='y', length=0, pad=4)
ax.set_xlabel('Change in median Spearman per section')
ax.text(XV / W, NR - 0.4 + 0.12, HEAD, fontsize=FS, color=NEUTRAL, transform=vt, va='bottom', ha='left')
YA_BOT = YA + NR * RH
hs = [Line2D([], [], marker='o', ls='none', mfc='#B5B5B5', mec='none', ms=3.8, label='Section'),
      Line2D([], [], marker='|', ls='none', mec=INK, ms=8, mew=1.8, label='Mean over sections'),
      Line2D([], [], marker='D', ls='none', mfc='white', mec=INK, ms=4.6, mew=0.9, label='Kidney held-out stripes')]
fig.legend(handles=hs, loc='upper left', bbox_to_anchor=(0.0, 1 - (YA_BOT + 0.47) / H), ncol=3, fontsize=FS, handletextpad=0.3,
           columnspacing=1.6, borderaxespad=0, borderpad=0)

# ---- bottom row: b (left) and c (right) share plot top and height ----
Y2 = YA_BOT + 0.80                      # panel letters of b and c
YP, PH = Y2 + 0.50, 1.45                # plot top, plot height (the key of b sits between its title and plot)
XB, WB = 0.42, 2.80                     # b: six groups at 0.47 in
XCL = 3.47                              # c: panel letter
XC = 4.10; WC = W - 0.08 - XC

# (b) stage 2 image statistics
R = pd.read_csv(RCSV)
R = R[R.method.isin(['mean (stage 1)', 'refined (constrained)']) & R.config.isin(['-', 'noise0.85'])]
M = R.groupby('method')[['sharp', 'hfpow', 'sd', 'cell_r', 'pix_r', 'within_r']].median()
label(fig, 0.0, Y2, 'b'); fig.text(0.17 / W, 1 - (Y2 + 0.01) / H, 'Stage 2, held-out kidney', fontsize=FS, va='top', color=INK)
ax = axB = axes_in(fig, XB, YP, WB, PH)
keys = [('sharp', 'Sharp-\nness'), ('hfpow', 'High-\nfreq.'), ('sd', 'Pixel\nSD'), ('cell_r', 'Cell\nr'), ('pix_r', 'Pixel\nr'), ('within_r', 'Within-\ncell r')]
x = np.arange(len(keys)); w = 0.38
for i, (m, col, lab) in enumerate([('mean (stage 1)', HERO, 'Stage 1 (mean)'), ('refined (constrained)', HERO_LIGHT, 'Stage 2 (refined)')]):
    ax.bar(x + (i - 0.5) * w, [M.loc[m, k] for k, _ in keys], w, color=col, label=lab, edgecolor='white', linewidth=0.3)
ax.axhline(1, color=INK, lw=0.7, ls=(0, (3, 1.5))); ax.axvline(2.5, color='#DDDDDD', lw=0.6)
ax.set_xticks(x, [k[1] for k in keys], linespacing=LS); ax.tick_params(axis='x', length=0, pad=3)
ax.set_xlim(-0.5, len(keys) - 0.5); ax.set_ylim(0, 1.4); ax.set_yticks([0, 0.5, 1.0])
ax.text(1, 1.16, 'Ratio to measured', ha='center', va='bottom', fontsize=FS, color=NEUTRAL)
ax.text(4, 1.16, 'Correlation', ha='center', va='bottom', fontsize=FS, color=NEUTRAL)
hb, lb = ax.get_legend_handles_labels()
fig.legend(hb, lb, loc='upper left', bbox_to_anchor=(0.17 / W, 1 - (Y2 + 0.25) / H), ncol=2, fontsize=FS,
           handlelength=1.0, columnspacing=1.2, borderaxespad=0, borderpad=0)

# (c) radial power spectra of a held-out window
Z = np.load(NPZ, allow_pickle=True)


def radial(a):
    a = a - a.mean(); P = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2; n = a.shape[0]
    yy, xx = np.indices(P.shape); r = np.hypot(yy - n // 2, xx - n // 2).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.bincount(r.ravel())


label(fig, XCL, Y2, 'c'); fig.text((XCL + 0.17) / W, 1 - (Y2 + 0.01) / H, 'Power spectrum (mean over proteins)', fontsize=FS, va='top',
                                   color=INK)
ax = axC = axes_in(fig, XC, YP, WC, PH)
n = Z['measured'].shape[1]; f = np.arange(n // 2) / n / 0.85  # cycles per µm
spec, ref = {}, None
for k, col, lab, ls in [('measured', INK, 'Measured', '-'), ('mean', HERO, 'Stage 1', '-'), ('refined', HERO_LIGHT, 'Stage 2', (0, (3, 1.5)))]:
    sp = np.mean([radial((Z[k][j] - Z[k][j].mean()) / (Z[k][j].std() + 1e-6))[:n // 2] for j in range(Z[k].shape[0])], 0)
    ref = sp[1] if ref is None else ref  # the measured image comes first
    assert np.all(sp[1:] > 0), 'radial power must be strictly positive before log10'
    ax.semilogx(f[1:], 10 * np.log10(sp[1:] / ref), color=col, lw=1.3, ls=ls, label=lab); spec[lab] = sp[1:]
ax.set_xticks([0.01, 0.03, 0.1, 0.3], ['0.01', '0.03', '0.1', '0.3']); ax.minorticks_off()
ax.set_yticks([0, -20, -40])
ax.set_xlabel('Spatial frequency (cycles per µm)', labelpad=2); ax.set_ylabel('Power (dB)', labelpad=2)
ax.legend(fontsize=FS, loc='lower left', handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)
save(fig, OUT, {'a_ablation': pd.DataFrame(out).T.round(4), 'b_stage2': M.round(4),
                'c_spectra': pd.DataFrame(spec, index=np.round(f[1:], 5)).round(6)},
     panels={'a': axA, 'b': axB, 'c': axC}, row_groups=[['b', 'c']])
print(pd.DataFrame(out).T.round(3).to_string()); print(M.round(3).to_string())
