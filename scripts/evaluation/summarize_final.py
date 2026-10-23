"""Per-dataset median Spearman (over markers) for every method found in results/bench/*/bench_rows.csv
(txcellnet.bench.bench_rows), split into kidney / development / test / fresh sections, with group means and paired wins of
a reference method. The per-dataset table (OUT_CSV with suffix _per_dataset) is the input of figures/figS1_ablation.py.
usage: summarize_final.py OUT_CSV REF_METHOD METHOD[,METHOD...]
"""
import sys
import numpy as np, pandas as pd

from txcellnet.bench import bench_rows

OUT, REF, METHODS = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
DEV = ['xen_lung_v1', 'xen_pancreas_v1', 'xen_breast_S1_Top', 'xen_breast_S1_Mid', 'xen_breast_S1_Bot', 'crlm_tumor1', 'crlm_tumor3_A1',
       'duchini_core01_hcc', 'vhd_lung_ffpe']
TEST = [f'xen_breast_S{s}_{p}' for s in (2, 3, 4) for p in ('Top', 'Mid', 'Bot')] + ['xen_lymphnode_5k', 'crlm_tumor2', 'crlm_tumor3_A2',
        'crlm_tumor3_A3', 'crlm_adjnormal', 'duchini_core02_non_tumour', 'duchini_core03_tonsil', 'duchini_core04_hca', 'vhd_breast_ffpe', 'vhd_tonsil_ff']
FRESH = ['xen_skin_5k', 'xen_prostate_5k', 'xen_ovary_v1ff', 'xen_ovary_5k', 'xen_breast_5k', 'xen_gbm_v1', 'xen_cervix_5k']
KID = ['kidney_rcc', 'kidney_ccrcc']
R = bench_rows()
T = R.groupby(['case', 'method']).spearman.median().unstack('method')
ms = [m for m in [REF] + METHODS if m in T]
miss = [m for m in [REF] + METHODS if m not in T]
if miss: print('missing methods:', miss)
rows = []
for split, cs in [('kidney', KID), ('dev', DEV), ('test', TEST), ('fresh', FRESH)]:
    cs = [c for c in cs if c in T.index]
    if not cs: continue
    sub = T.loc[cs, ms]
    print(f'\n== {split} ({len(cs)} datasets)\n' + sub.round(3).to_string())
    mean = sub.mean(); rows.append(dict(split=split, n=len(cs), **mean.round(4).to_dict()))
    if REF in sub:
        print('mean:', ' '.join(f'{m}={v:.3f}' for m, v in mean.items()))
        print('wins of', REF, ':', ' '.join(f'{m}={int((sub[REF] > sub[m]).sum())}/{int(sub[m].notna().sum())}' for m in ms if m != REF))
pd.DataFrame(rows).to_csv(OUT, index=False)
T.to_csv(OUT.replace('.csv', '_per_dataset.csv'))
