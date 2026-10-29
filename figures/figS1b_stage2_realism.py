"""Stage 2 vs stage 1: which image is closer to the measured one? (RECOMB 10 pt floor; 6.5 x 2.2 in, printed size)

Kidney held-out stripes, 27 IF proteins; stage 1 = 6-network ensemble mean, stage 2 = refined image (tau 0.85, 20 Euler
steps, one sample). Per protein, the distance of each stage to the measured image is averaged over the two sections (RCC,
ccRCC); each dot is one protein's ratio distance(stage 2) / distance(stage 1) (log axis; < 1 = stage 2 closer, dashed line =
stage 1); bar = median over proteins; open grey diamond = median ratio of the matched-noise control (stage 1 + Gaussian
noise with the SD of the stage-2 residual, cell means kept), distance(control) / distance(stage 1); right column = proteins
where stage 2 is closer than stage 1 (of 27).
Distances (scripts/evaluation/stage2_realism.py, stage2_kid_cellpose.py):
  whole image  pixel intensity: 1-Wasserstein distance of held-out pixel values; GLCM (Haralick) contrast, homogeneity,
               energy, correlation at 3 px on 48 windows of 256 x 256 px, 32 grey levels fixed by the measured image:
               |log(feature / measured)|; patch SWD: sliced Wasserstein distance of 7 x 7 Laplacian-pyramid patches;
               KID on Cellpose 'nuclei' style vectors (192 tiles of 128 x 128 px)
  per cell     KS distance between the distributions over held-out cells of within-cell intensity SD, CV, edge / centre
               intensity ratio and fine granularity (fraction of the cell's intensity removed by a grey opening with a
               disk of radius 1 px: first step of the granular spectrum), linear intensity
  correlation  1 - Pearson r with the measured image over held-out pixels, and within cells (pixel minus its cell mean)
usage: figS1b_stage2_realism.py PER_PROTEIN_ALL_CSV OUT_PREFIX
"""
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D
import figstyle
figstyle.print_min10()
from figstyle import W, axes_in, save, INK, HERO, HERO_LIGHT, NEUTRAL

FS = 10
CSV, OUT = sys.argv[1:3]
D = pd.read_csv(CSV)
assert set(D.section) == {'rcc', 'ccrcc'}, 'both kidney sections expected'
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
        g = D[D.metric == m].groupby('protein')[['dist_stage1', 'dist_stage2', 'dist_noise']].mean()
        assert len(g) == 27 and g.notna().all().all(), m
        r = g.dist_stage2 / g.dist_stage1; rn = g.dist_noise / g.dist_stage1; ratio[m], nratio[m] = r, rn
        out.append(dict(metric=m, label=lab, median_dist_stage1=g.dist_stage1.median(), median_dist_stage2=g.dist_stage2.median(),
                        median_dist_noise=g.dist_noise.median(), median_ratio=r.median(), median_ratio_noise=rn.median(),
                        n_stage2_closer=int((r < 1).sum()), n_stage2_closer_than_noise=int((g.dist_stage2 < g.dist_noise).sum()),
                        n=len(r)))
allr = pd.concat(ratio.values()); nmed = pd.Series({m: v.median() for m, v in nratio.items()})
XL = (0.018, 10.0)
assert allr.min() > XL[0] and allr.max() < XL[1], f'ratios outside the axis: {allr.min():.3f}..{allr.max():.3f}'
assert nmed.min() > XL[0] and nmed.max() < XL[1], f'control medians outside the axis: {nmed.min():.3f}..{nmed.max():.3f}'

H = 2.2
fig = plt.figure(figsize=(W, H)); rend = fig.canvas.get_renderer()


def width_in(s):
    t = fig.text(0, 0, s, fontsize=FS); w = t.get_window_extent(rend).width / fig.dpi; t.remove(); return w


LABW = [max(width_in(lab) for _, lab in BLOCKS[0][2]), max(width_in(lab) for _, _, rows in BLOCKS[1:] for _, lab in rows)]
CNTW = max(width_in('27/27'), width_in('Closer'))
YP, RH = 0.34, 0.2                    # plot top, row pitch (in)
GL, GC, GB = 0.10, 0.08, 0.30         # label-plot gap, plot-count gap, gap between the columns
PW = (W - 0.02 - sum(LABW) - 2 * (GL + GC + CNTW) - GB) / 2   # equal plot widths filling the figure width
XA = [LABW[0] + GL, LABW[0] + GL + PW + GC + CNTW + GB + LABW[1] + GL]   # plot left edges of the two columns
NB = len(BLOCKS[1][2])
FRAME = {'a': (0, YP, NR), 'b': (1, YP, NB), 'c': (1, YP + (NB + 1) * RH, NR - NB - 1)}   # column, top (in), rows
axes = {}
for bid, title, rows in BLOCKS:
    col, top, nr = FRAME[bid]; xa = XA[col]; x0 = xa - GL - LABW[col]
    ax = axes_in(fig, xa, top, PW, nr * RH); axes[bid] = ax
    if top == YP:                     # block title: top line of the figure, or centred in the free row above the block
        fig.text(x0 / W, 1 - 0.03 / H, title, fontsize=FS, va='top', ha='left', color=INK)
    else:
        fig.text(x0 / W, 1 - (top - RH / 2) / H, title, fontsize=FS, va='center_baseline', ha='left', color=INK)
    vt = blended_transform_factory(fig.transFigure, ax.transData)
    rng = np.random.default_rng(0)
    for i, (m, lab) in enumerate(rows):
        y = nr - 1 - i; r = ratio[m].values
        ax.scatter(r, y + rng.uniform(-0.25, 0.25, len(r)), s=8, c=HERO_LIGHT, edgecolors='none', zorder=3)
        ax.plot([np.median(r)] * 2, [y - 0.38, y + 0.38], color=HERO, lw=1.8, solid_capstyle='butt', zorder=4)
        ax.plot(nmed[m], y, marker='D', ms=4.6, mfc='white', mec=NEUTRAL, mew=1.0, ls='none', zorder=5)
        ax.text((xa + PW + GC) / W, y, f'{int((r < 1).sum())}/{len(r)}', fontsize=FS, va='center', ha='left', transform=vt, color=INK)
    ax.set_xscale('log'); ax.set_xlim(*XL); ax.set_ylim(-0.5, nr - 0.5)
    ax.set_yticks(range(nr), [lab for _, lab in rows][::-1], fontsize=FS)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.axvline(1, color=INK, lw=0.7, ls=(0, (3, 1.5)), zorder=2)
    if top == YP:
        ax.text(1, nr - 0.5, 'Stage 1', fontsize=FS, color=NEUTRAL, ha='center', va='bottom')
        ax.text((xa + PW + GC) / W, nr - 0.5, 'Closer', fontsize=FS, color=NEUTRAL, ha='left', va='bottom', transform=vt)
    if bid == 'b':                    # the correlation block below carries the x axis of the right column
        ax.spines['bottom'].set_visible(False); ax.set_xticks([]); ax.minorticks_off()
    else:
        ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"]); ax.set_xticks([0.03, 0.3, 3], minor=True)
        ax.tick_params(axis='x', which='minor', labelbottom=False)
        ax.set_xlabel('Distance ratio, stage 2 / stage 1', labelpad=2)
# key of the control marker: top line, right-aligned with the left column's count column
kx = XA[0] + PW + GC + CNTW
t = fig.text(kx / W, 1 - 0.03 / H, 'Stage 1 + matched noise', fontsize=FS, va='top', ha='right', color=INK)
tw = t.get_window_extent(rend).width / fig.dpi
fig.add_artist(Line2D([(kx - tw - 0.09) / W], [1 - (0.03 + 0.075) / H], marker='D', ms=4.6, mfc='white', mec=NEUTRAL, mew=1.0,
                      ls='none', transform=fig.transFigure))
src = pd.DataFrame({m: ratio[m] for m in ratio}).round(5)
srcn = pd.DataFrame({m: nratio[m] for m in nratio}).round(5)
save(fig, OUT, {'ratios': src, 'ratios_noise': srcn, 'summary': pd.DataFrame(out).set_index('metric').round(4)},
     panels=axes, column_groups=[['b', 'c']])
print(pd.DataFrame(out).set_index('metric').round(3).to_string())
