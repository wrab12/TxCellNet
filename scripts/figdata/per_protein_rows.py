"""Data of the per-protein panel of Fig. 2b: the benchmark rows (one per case, marker and method) of the kidney and CRLM
sections, for the models of Supplementary Tables S3 (cross-dataset setting) and S4 (same-section setting) and the
cross-patient kidney networks of Supplementary Fig. S2. Rows of results/bench/*/bench_rows.csv (txcellnet.bench.bench_rows).
usage: per_protein_rows.py OUT_DIR   -> OUT_DIR/per_protein/bench_rows_kidney_crlm.csv
"""
import os, sys

from txcellnet.bench import bench_rows

OUT = os.path.join(sys.argv[1], 'per_protein'); os.makedirs(OUT, exist_ok=True)
METHODS = ['txcellnet', 'fin_clean_s1', 'nichetrans_off_u', 'nichetrans_off_e3', 'nichetrans_off', 'dgat_pm', 'spamosaic_pm',
           'totalvi_pm', 'mlp_pm_u', 'mlp_pm_e3', 'mlp_pm', 'own_mRNA', 'x_ccrcc_only', 'x_rcc_only', 'ridge_pm_u', 'ridge_pm',
           'knn_pm_u', 'knn_pm', 'celltype_pm_u', 'celltype_pm']
R = bench_rows()
R = R[(R.case.str.startswith('kidney') | R.case.str.startswith('crlm')) & R.method.isin(METHODS)]
R.to_csv(f'{OUT}/bench_rows_kidney_crlm.csv', index=False)
print(R.groupby(['case', 'method']).size().unstack('case').to_string()); print('saved', f'{OUT}/bench_rows_kidney_crlm.csv')
