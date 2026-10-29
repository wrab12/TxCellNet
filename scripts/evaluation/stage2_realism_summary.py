"""Summary of the stage-2 realism metrics (stage2_realism.py + stage2_kid_cellpose.py): per metric and section, median over
the 27 proteins of stage 1, stage 2 and measured values and of the distances to measured, number of proteins where stage 2
is closer, and a paired one-sided Wilcoxon signed-rank test over proteins (H1: stage 2 closer to measured than stage 1).
'mean of sections' = per-protein mean of the two sections (n = 27). Correlations (pix_r, within_r): two-sided test of the
change, 'closer' = higher correlation.
Matched-noise controls (columns dist_noise, dist_noise_s1, when present): median distance of the control, number of proteins
where stage 2 is closer than the control, and the same paired test of stage 2 against the control.
usage: stage2_realism_summary.py PER_PROTEIN_CSV [KID_CSV ...] OUT_CSV
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

*ins, OUT = sys.argv[1:]
D = pd.concat([pd.read_csv(f) for f in ins], ignore_index=True)
D.to_csv(OUT.replace('_summary.csv', '_per_protein_all.csv'), index=False)
CTRL = [k for k in ['noise', 'noise_s1'] if f'dist_{k}' in D]


def paired(d2, d1, alt):
    ok = ~(np.isnan(d1) | np.isnan(d2))
    p = wilcoxon(d2[ok], d1[ok], alternative=alt).pvalue if ok.sum() > 5 and np.any(d2[ok] != d1[ok]) else np.nan
    return ok, p


S = []
for (metric, fam), g in D.groupby(['metric', 'family'], sort=False):
    if g.dist_stage1.isna().all():   # descriptive rows without a distance (control noise SD): medians only
        for sec, h in list(g.groupby('section')) + [('mean of sections', g.groupby('protein')[['stage2'] + CTRL].mean().reset_index())]:
            S.append(dict(metric=metric, family=fam, section=sec, n=len(h), **{f'median_{k}': float(np.median(h[k])) for k in CTRL}))
        continue
    cols = ['stage1', 'stage2', 'measured', 'dist_stage1', 'dist_stage2'] + [c for k in CTRL for c in (k, f'dist_{k}')]
    for sec, h in list(g.groupby('section')) + [('mean of sections', g.groupby('protein')[cols].mean().reset_index())]:
        d1, d2 = h.dist_stage1.values.astype(float), h.dist_stage2.values.astype(float)
        alt = 'two-sided' if fam == 'correlation' else 'less'
        ok, p = paired(d2, d1, alt)
        r = dict(metric=metric, family=fam, section=sec, n=int(ok.sum()),
                 median_measured=float(np.median(h.measured)) if h.measured.notna().any() else np.nan,
                 median_stage1=float(np.median(h.stage1)), median_stage2=float(np.median(h.stage2)),
                 median_dist_stage1=float(np.median(d1[ok])), median_dist_stage2=float(np.median(d2[ok])),
                 n_stage2_closer=int((d2[ok] < d1[ok]).sum()), wilcoxon_p=p, test=f'paired Wilcoxon, {alt}')
        for k in CTRL:
            dn = h[f'dist_{k}'].values.astype(float); okn, pn = paired(d2, dn, alt)
            r.update({f'median_dist_{k}': float(np.median(dn[okn])), f'n_stage2_closer_than_{k}': int((d2[okn] < dn[okn]).sum()),
                      f'wilcoxon_p_stage2_vs_{k}': pn})
        S.append(r)
S = pd.DataFrame(S); S.to_csv(OUT, index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(S[S.section == 'mean of sections'].drop(columns=['family', 'test', 'section'] + [c for c in S if c.endswith('noise_s1')])
      .to_string(index=False, float_format=lambda v: f'{v:.4g}'))
