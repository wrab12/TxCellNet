"""Supplementary Table S9 (case study): agreement with IMC per tumour section and source, from crlm_case.py's summary.csv;
median vocabulary molecules per scored cell from own_only.py's bench_cases.csv (column median_vocab_tx_per_cell).
usage: case_table.py SUMMARY_CSV BENCH_CASES_CSV OUT_TEX"""
import sys
import numpy as np, pandas as pd

S = pd.read_csv(sys.argv[1]); st = pd.read_csv(sys.argv[2]).set_index('case'); out = sys.argv[3]
SECTIONS = [('crlm_tumor1', 'Tumour 1'), ('crlm_tumor2', 'Tumour 2'), ('crlm_tumor3_A1', 'Tumour 3, A1'), ('crlm_tumor3_A2', 'Tumour 3, A2'),
            ('crlm_tumor3_A3', 'Tumour 3, A3')]
SRC = ['TxCellNet', 'DGAT', 'NicheTrans', 'Cognate mRNA']
MET = [('dice_tumour', 'Tumour region (Dice)'), ('r_immune', 'CD45$^+$ profile ($r$)'), ('r_macrophage', 'CD68$^+$ profile ($r$)')]
fmt = lambda v: f'{v:.2f}'.replace('-', '$-$')
L = [r'\begin{tabular}{@{}lr' + 'rrrr' * len(MET) + r'c@{}}', r'\toprule',
     r' & & ' + ' & '.join(rf'\multicolumn{{4}}{{c}}{{{t}}}' for _, t in MET) + r' & IMC immune \\',
     ''.join(rf'\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}' for i in range(len(MET))),
     r'Section & Molec. & ' + ' & '.join(['TxC & DGAT & NT & mRNA'] * len(MET)) + r' & in / out \\', r'\midrule']
for c, name in SECTIONS:
    d = S[S.case == c].set_index('source'); cells = []
    for m, _ in MET:
        v = d.loc[SRC, m]; best = fmt(v.max())
        cells += [rf'\textbf{{{fmt(x)}}}' if fmt(x) == best else fmt(x) for x in v]
    imc = d.loc['IMC']
    L.append(f'{name} & {int(st.loc[c, "median_vocab_tx_per_cell"])} & ' + ' & '.join(cells) +
             f' & {imc.immune_in:.2f} / {imc.immune_out:.2f} \\\\')
mean = S[S.source.isin(SRC)].groupby('source')[[m for m, _ in MET]].mean()
cells = []
for m, _ in MET:
    v = mean.loc[SRC, m]; best = fmt(v.max()); cells += [rf'\textbf{{{fmt(x)}}}' if fmt(x) == best else fmt(x) for x in v]
L += [r'\midrule', 'Mean & & ' + ' & '.join(cells) + r' & \\', r'\bottomrule', r'\end{tabular}']
open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
