"""Localization part of Supplementary Table S1 (semi-synthetic benchmark): one row per synthetic channel, mean of the two
sections, from the rows of sim_loc.py (results/sim/loc/sim_loc_rows.csv; local copy data/allm/sim_loc/sim_loc_rows.csv).
TxCellNet = clean + randomized networks (stage 1, conditional mean). Columns: area share of the intended compartment,
share of the cell signal in the intended compartment (truth, TxCellNet, range of the six per-cell methods), fraction of
expressing cells whose prediction is higher in the intended compartment than in the rest of the cell (truth, TxCellNet;
0 for every per-cell method by construction), within-cell Pearson of TxCellNet.
Writes OUT_PREFIX.tex (English), OUT_PREFIX_zh.tex (Chinese) and OUT_PREFIX_data.csv (the values, 4 decimals); prints the
numbers quoted in the text. Meant to be set below the method table of Table S1, inside the same table float.
usage: sim_loc_table.py SIM_LOC_ROWS_CSV OUT_PREFIX
"""
import sys
import pandas as pd

L = pd.read_csv(sys.argv[1]); OUT = sys.argv[2]
CH = ['mem_immune', 'nuc_prolif', 'cyto_epith', 'niche_r15', 'niche_r40', 'niche_r80']
TX = 'TxCellNet:ens_cr'
PERCELL = ['NicheTrans', 'DGAT', 'SpaMosaic', 'totalVI', 'ridge_niche', 'ridge']
assert set(PERCELL + [TX, 'truth']) <= set(L.method) and L.groupby(['method', 'channel']).section.nunique().eq(2).all()
M = L.groupby(['method', 'channel'])[['share_in', 'correct_cells', 'within_cell', 'area_share']].mean()
T = pd.DataFrame({
    'compartment': L.drop_duplicates('channel').set_index('channel').compartment.reindex(CH),
    'area_share': [M.loc[('truth', c), 'area_share'] for c in CH],
    'truth_share': [M.loc[('truth', c), 'share_in'] for c in CH],
    'txcellnet_share': [M.loc[(TX, c), 'share_in'] for c in CH],
    'percell_share_min': [min(M.loc[(m, c), 'share_in'] for m in PERCELL) for c in CH],
    'percell_share_max': [max(M.loc[(m, c), 'share_in'] for m in PERCELL) for c in CH],
    'truth_correct_cells': [M.loc[('truth', c), 'correct_cells'] for c in CH],
    'txcellnet_correct_cells': [M.loc[(TX, c), 'correct_cells'] for c in CH],
    'percell_correct_cells_max': [max(M.loc[(m, c), 'correct_cells'] for m in PERCELL) for c in CH],
    'txcellnet_within_cell': [M.loc[(TX, c), 'within_cell'] for c in CH]}, index=pd.Index(CH, name='channel'))
assert T.percell_correct_cells_max.eq(0).all()   # a uniform fill is never higher in one compartment than in the rest
T.round(4).to_csv(f'{OUT}_data.csv')

pc = lambda v: f'{100 * v:.0f}'
pc1 = lambda v: f'{100 * v:.1f}'
NAME = {'en': {'mem_immune': 'Immune program', 'nuc_prolif': 'Proliferation program', 'cyto_epith': 'Epithelial program',
               'niche_r15': r'Induced, 15\,$\mu$m', 'niche_r40': r'Induced, 40\,$\mu$m', 'niche_r80': r'Induced, 80\,$\mu$m'},
        'zh': {'mem_immune': '免疫程序', 'nuc_prolif': '增殖程序', 'cyto_epith': '上皮程序',
               'niche_r15': r'邻域诱导，15\,$\mu$m', 'niche_r40': r'邻域诱导，40\,$\mu$m', 'niche_r80': r'邻域诱导，80\,$\mu$m'}}
COMP = {'en': {'ring': 'membrane ring', 'nuc': 'nucleus', 'cyto': 'cytoplasm'},
        'zh': {'ring': '膜环', 'nuc': '细胞核', 'cyto': '细胞质'}}
HEAD = {'en': [r' & & & \multicolumn{3}{c}{signal in compartment (\%)} & \multicolumn{2}{c}{cells correct (\%)} & \\',
               r'Channel & Compartment & area (\%) & truth & TxCellNet & per-cell & truth & TxCellNet & within-cell $r$\\'],
        'zh': [r' & & & \multicolumn{3}{c}{区室信号比例（\%）} & \multicolumn{2}{c}{区室正确率（\%）} & \\',
               r'通道 & 预定区室 & 面积（\%） & 真值 & TxCellNet & 逐细胞方法 & 真值 & TxCellNet & 细胞内 Pearson\\']}
for lang, suf in (('en', ''), ('zh', '_zh')):
    S = [r'\begin{tabular}{@{}llccccccc@{}}', r'\toprule', HEAD[lang][0], r'\cmidrule(lr){4-6}\cmidrule(lr){7-8}', HEAD[lang][1],
         r'\midrule']
    for c, r in T.iterrows():
        S.append(' & '.join([NAME[lang][c], COMP[lang][r.compartment], pc(r.area_share), pc(r.truth_share),
                             pc(r.txcellnet_share), f'{pc(r.percell_share_min)}--{pc(r.percell_share_max)}',
                             pc1(r.truth_correct_cells), pc1(r.txcellnet_correct_cells), f'{r.txcellnet_within_cell:.2f}']) + r'\\')
    S += [r'\bottomrule', r'\end{tabular}']
    open(f'{OUT}{suf}.tex', 'w').write('\n'.join(S) + '\n')
print(open(f'{OUT}_zh.tex').read())
print(T.round(4).to_string())
print('TxCellNet cells correct, mean of the two sections: %.4f-%.4f' % (T.txcellnet_correct_cells.min(), T.txcellnet_correct_cells.max()))
X = L[L.method == TX]
print('per section: %.4f-%.4f' % (X.correct_cells.min(), X.correct_cells.max()))
print('area share %.3f-%.3f, per-cell share %.3f-%.3f' % (T.area_share.min(), T.area_share.max(), T.percell_share_min.min(),
                                                            T.percell_share_max.max()))
