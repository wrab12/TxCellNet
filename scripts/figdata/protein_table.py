"""Data of Supplementary Fig. S2 (which proteins transfer): TxCellNet's cell-level Spearman per protein in four settings,
kidney held-out stripes (six-network ensemble 'txcellnet') | kidney cross-patient (one mixed single-section network per
section, 'x_<section>_only', trained on one section reading the genes measured in both, scored on the other) | CRLM, IMC (test
sections, median) | liver / tonsil, COMET (test cores, median); plus the cognate mRNA on CRLM and TxCellNet minus NicheTrans on CRLM.
Rows of results/bench/*/bench_rows.csv (txcellnet.bench.bench_rows).
usage: protein_table.py OUT_DIR [NICHETRANS_METHOD=nichetrans_off_e3]   -> OUT_DIR/protein_table.csv
"""
import os, sys
import numpy as np, pandas as pd

from txcellnet.bench import bench_rows

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
NT = sys.argv[2] if len(sys.argv) > 2 else 'nichetrans_off_e3'; TX = 'txcellnet'
R = bench_rows()

CAT = {'immune lineage': ['CD20', 'CD3E', 'CD4', 'CD8A', 'CD45', 'CD45RO', 'CD45RA', 'CD68', 'CD163', 'CD11c', 'CD16'],
       'epithelial / tumour': ['PanCK', 'E-Cadherin', 'Beta-catenin'],
       'stromal / vascular': ['Vimentin', 'alphaSMA', 'CD31'],
       'cell state / checkpoint': ['Ki-67', 'PD-1', 'PD-L1', 'VISTA', 'GranzymeB']}
ORDER = [p for v in CAT.values() for p in v]
CRLM_T = ['crlm_tumor2', 'crlm_tumor3_A2', 'crlm_tumor3_A3', 'crlm_adjnormal']
COMET_T = ['duchini_core02_non_tumour', 'duchini_core03_tonsil', 'duchini_core04_hca']


def med(cs, method, marker):
    g = R[R.case.isin(cs) & (R.method == method) & (R.marker == marker)]
    return g.spearman.median() if len(g) else np.nan


S = {}
for p in ORDER:
    S[p] = {'kidney\nheld-out': np.nanmean([med(['kidney_rcc'], TX, p), med(['kidney_ccrcc'], TX, p)]),
            'kidney\ncross-patient': np.nanmean([med(['kidney_rcc'], 'x_ccrcc_only', p), med(['kidney_ccrcc'], 'x_rcc_only', p)]),
            'CRLM\n(IMC)': med(CRLM_T, TX, p),
            'tonsil / liver\n(COMET)': med(COMET_T, TX, p),
            'CRLM\ncognate mRNA': med(CRLM_T, 'own_mRNA', p),
            'CRLM\nminus NicheTrans': med(CRLM_T, TX, p) - med(CRLM_T, NT, p)}
H = pd.DataFrame(S).T
H.to_csv(f'{OUT}/protein_table.csv')
print(H.round(2).to_string()); print('saved', f'{OUT}/protein_table.csv')
