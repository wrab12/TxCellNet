"""Supplementary Table S13 (appended after Table S12, so the existing table numbers stay): cell-level Spearman of every
protein and method on the two datasets of main Fig. 2b, with values. Same numbers as Fig. 2b (perprotein.py), all proteins:
kidney held-out stripes, same-section setting (27 proteins; models of Table S4; mean of the two sections) and CRLM IMC,
kidney-trained (22 proteins; models of Table S3; median of the four test sections). Best per row and dataset in
bold (ties at two decimals all bold); -- = protein not measured in that dataset, or no cognate transcript.
Writes OUT_PREFIX.tex (English), OUT_PREFIX_zh.tex (Chinese) and OUT_PREFIX_data.csv (the values, 4 decimals).
usage: table_per_protein.py DATA_DIR OUT_PREFIX
"""
import sys
import numpy as np, pandas as pd
from perprotein import METHODS, CAT, PRETTY, per_protein

DD, OUT = sys.argv[1:3]
PK, PCR = per_protein(DD)
HEAD = {'en': ['TxC', 'NT', 'DGAT', 'SpaM.', 'totalVI', 'MLP', 'mRNA'], 'zh': ['TxC', 'NT', 'DGAT', 'SpaM.', 'totalVI', 'MLP', 'mRNA']}
BLOCK = {'en': ('Kidney, same section (27)', 'CRLM, kidney-trained (22)'), 'zh': ('肾脏，同一切片（27 种）', 'CRLM，只在肾脏上训练（22 种）')}
CATN = {'en': [c for c, _ in CAT], 'zh': ['免疫谱系', '上皮与肿瘤', '基质与血管', '细胞状态与检查点']}
FIRST = {'en': 'Protein', 'zh': '蛋白'}
TEX = {'β-catenin': '$\\beta$-catenin', 'αSMA': '$\\alpha$SMA'}


def cells(v):
    """formatted values of one row of one dataset, best in bold (every value equal to the best at two decimals); --
    for missing"""
    top = f'{v.max():.2f}' if v.notna().any() else None
    out = []
    for m in METHODS:
        if not np.isfinite(v[m]): out.append('--'); continue
        s = f'{v[m]:.2f}'.replace('-0.00', '0.00')
        out.append((f'\\textbf{{{s}}}' if s == top else s).replace('-', '$-$'))
    return out


def table(lang):
    n = 1 + 2 * len(METHODS)
    # 15 columns fit the 6.5 in text width at \\small (10 pt, the RECOMB minimum) with 1.2 pt column padding (set here, local to the table)
    L = ['\\small\\setlength{\\tabcolsep}{1.2pt}%',
         '\\begin{tabular}{@{}l' + 'r' * len(METHODS) + '@{\\hspace{4pt}}' + 'r' * len(METHODS) + '@{}}', '\\toprule',
         f' & \\multicolumn{{{len(METHODS)}}}{{c}}{{{BLOCK[lang][0]}}} & \\multicolumn{{{len(METHODS)}}}{{c}}{{{BLOCK[lang][1]}}}\\\\',
         f'\\cmidrule(lr){{2-{1 + len(METHODS)}}}\\cmidrule(l){{{2 + len(METHODS)}-{n}}}',
         FIRST[lang] + ' & ' + ' & '.join(HEAD[lang] * 2) + '\\\\', '\\midrule']
    for k, (_, prots) in enumerate(CAT):
        if k: L.append('\\midrule')
        L.append(f'\\multicolumn{{{n}}}{{@{{}}l}}{{\\textit{{{CATN[lang][k]}}}}}\\\\')
        for p in prots:
            name = PRETTY.get(p, p)
            L.append(TEX.get(name, name) + ' & ' + ' & '.join(cells(PK.loc[p].astype(float)) + cells(PCR.loc[p].astype(float))) + '\\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(L) + '\n'


for lang, suf in (('en', ''), ('zh', '_zh')):
    with open(f'{OUT}{suf}.tex', 'w') as f: f.write(table(lang))
D = pd.concat({'kidney_same_section': PK, 'crlm_kidney_trained': PCR}, axis=1)
D.columns = [f'{a}_{b}' for a, b in D.columns]
D.round(4).to_csv(f'{OUT}_data.csv')
print(table('en'))
print('best per protein, kidney:', PK.dropna(how='all').idxmax(1).value_counts().to_dict(),
      ' CRLM:', PCR.dropna(how='all').idxmax(1).value_counts().to_dict())
