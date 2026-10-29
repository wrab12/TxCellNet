"""Per-section tables of the benchmark (Supplementary Tables S3, S4) and the per-section data of Fig. 2 and Supplementary
Fig. S5.

within  same-section setting: every method is trained on the training stripes of each section and scored on its held-out
        stripes (34 sections; kidney = the held-out stripes of the two kidney sections, models trained on both kidney sections)
cross   cross-dataset setting: every method is trained on the two kidney sections only and applied to the 17 test sections
        and the 7 sections added after the configuration freeze
Outputs (OUT_DIR): fig_<mode>_per_dataset.csv (median cell-level Spearman over markers, per section and method),
fig_<mode>_groups.csv (group means), fig_<mode>_stats.json (means, wins, ranks), table_<mode>.tex (supplementary table).
Sources: results/bench/within/*.csv (same-section setting), results/bench/*/bench_rows.csv (cross-dataset setting and the
kidney rows).
usage: bench_tables.py within|cross OUT_DIR
"""
import glob, json, os, sys
import numpy as np, pandas as pd

from txcellnet.bench import bench_rows

MODE, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
# key, label; published methods first, then baselines
METHODS = [('txcellnet', 'TxCellNet'), ('nichetrans', 'NicheTrans'), ('dgat', 'DGAT'), ('spamosaic', 'SpaMosaic'), ('totalvi', 'totalVI'),
           ('mlp', 'MLP'), ('ridge', 'Ridge'), ('knn', 'kNN mapping'), ('celltype', 'Cell-type lookup'), ('own_mRNA', 'Cognate mRNA')]
SHORT = {'kidney_rcc': 'RCC', 'kidney_ccrcc': 'ccRCC', 'duchini_core01_hcc': 'hepatocellular carcinoma', 'duchini_core02_non_tumour': 'normal liver',
         'duchini_core03_tonsil': 'tonsil', 'duchini_core04_hca': 'hepatocellular adenoma', 'crlm_tumor1': 'tumor 1', 'crlm_tumor2': 'tumor 2',
         'crlm_adjnormal': 'adjacent normal liver', 'crlm_tumor3_A1': 'tumor 3, ROI A1', 'crlm_tumor3_A2': 'tumor 3, ROI A2',
         'crlm_tumor3_A3': 'tumor 3, ROI A3', 'xen_lung_v1': 'lung cancer', 'xen_pancreas_v1': 'pancreas', 'xen_lymphnode_5k': 'reactive lymph node, 5K',
         'xen_skin_5k': 'melanoma, 5K', 'xen_prostate_5k': 'prostate cancer, 5K', 'xen_ovary_5k': 'ovarian cancer, 5K',
         'xen_ovary_v1ff': 'ovarian cancer, fresh frozen', 'xen_breast_5k': 'breast cancer, 5K', 'xen_cervix_5k': 'cervical cancer, 5K',
         'xen_gbm_v1': 'glioblastoma, IO panel'}
for s in (1, 2, 3, 4):
    for p, q in (('Top', 'top'), ('Mid', 'middle'), ('Bot', 'bottom')): SHORT[f'xen_breast_S{s}_{p}'] = f'sample {s}, {q}'
BREAST = lambda ss: [f'xen_breast_S{s}_{p}' for s in ss for p in ('Top', 'Mid', 'Bot')]
TEST = BREAST((2, 3, 4)) + ['xen_lymphnode_5k', 'crlm_tumor2', 'crlm_tumor3_A2', 'crlm_tumor3_A3', 'crlm_adjnormal', 'duchini_core02_non_tumour',
                            'duchini_core03_tonsil', 'duchini_core04_hca']
FRESH = ['xen_skin_5k', 'xen_prostate_5k', 'xen_ovary_5k', 'xen_ovary_v1ff', 'xen_breast_5k', 'xen_cervix_5k', 'xen_gbm_v1']
if MODE == 'within':
    GROUPS = [('Kidney cancer, Xenium Protein (27 proteins)', 'Kidney\n(IF)', ['kidney_rcc', 'kidney_ccrcc']),
              ('Colorectal liver metastasis, IMC (22 proteins)', 'CRLM\n(IMC)', ['crlm_tumor1', 'crlm_tumor2', 'crlm_adjnormal', 'crlm_tumor3_A1',
                                                                                'crlm_tumor3_A2', 'crlm_tumor3_A3']),
              ('Liver and tonsil cores, COMET (7 proteins)', 'Liver,\ntonsil\n(COMET)', ['duchini_core01_hcc', 'duchini_core02_non_tumour', 'duchini_core03_tonsil',
                                                                                      'duchini_core04_hca']),
              ('Breast cancer, Xenium 280-gene panel', 'Breast\n(stains)', BREAST((1, 2, 3, 4))),
              ('Lung cancer and pancreas, Xenium v1', 'Lung,\npancreas', ['xen_lung_v1', 'xen_pancreas_v1']),
              ('Xenium 5K and other panels', '5K,\nother', ['xen_lymphnode_5k', 'xen_skin_5k', 'xen_prostate_5k', 'xen_ovary_5k', 'xen_ovary_v1ff',
                                                                 'xen_breast_5k', 'xen_cervix_5k', 'xen_gbm_v1'])]
else:
    GROUPS = [('Breast cancer, 280-gene panel (73 shared genes)', 'Breast', BREAST((2, 3, 4))),
              ('Reactive lymph node, Xenium 5K', 'Lymph\nnode', ['xen_lymphnode_5k']),
              ('Colorectal liver metastasis, IMC (22 proteins)', 'CRLM\n(IMC)', ['crlm_tumor2', 'crlm_tumor3_A2', 'crlm_tumor3_A3', 'crlm_adjnormal']),
              ('Liver and tonsil cores, COMET (7 proteins)', 'Liver,\ntonsil', ['duchini_core02_non_tumour', 'duchini_core03_tonsil', 'duchini_core04_hca']),
              ('Fresh datasets (added after the design freeze)', 'Fresh', FRESH)]
ALL = [c for _, _, cs in GROUPS for c in cs]

R = bench_rows()
CROSS = R.groupby(['case', 'method']).spearman.median().unstack('method')
W = pd.concat([pd.read_csv(f) for f in glob.glob('results/bench/within/*.csv')]) if glob.glob('results/bench/within/*.csv') else pd.DataFrame(
    columns=['case', 'marker', 'method', 'spearman'])
WITHIN = W.groupby(['case', 'method']).spearman.median().unstack('method')


def pick(T, cols, cases):
    """per-dataset value of a method given as a column name or a list of columns (averaged, e.g. single networks)."""
    cols = [cols] if isinstance(cols, str) else cols
    have = [c for c in cols if c in T]
    return T.reindex(cases)[have].mean(1) if have else pd.Series(np.nan, index=cases)


TX_SINGLE = ['fin_clean_s1', 'fin_clean_s2', 'fin_clean_s3', 'fin_rand_s1', 'fin_rand_s2', 'fin_rand_s3']
WMAP = {'txcellnet': 'within:within_clean', 'nichetrans': 'within:nichetrans_off', 'dgat': 'within:dgat', 'spamosaic': 'within:spamosaic',
        'totalvi': 'within:totalvi', 'mlp': 'within:mlp', 'ridge': 'within:ridge', 'knn': 'within:knn', 'celltype': 'within:celltype',
        'own_mRNA': 'within:own_mRNA'}
# kidney held-out stripes = within-dataset for the training tissue: one clean network; baselines read all panel genes ('_u')
KMAP = {'txcellnet': 'fin_clean_s1', 'nichetrans': 'nichetrans_off_u', 'dgat': 'dgat_pm', 'spamosaic': 'spamosaic_pm', 'totalvi': 'totalvi_pm',
        'mlp': 'mlp_pm_u', 'ridge': 'ridge_pm_u', 'knn': 'knn_pm_u', 'celltype': 'celltype_pm_u', 'own_mRNA': 'own_mRNA'}
# cross-dataset: TxCellNet = the frozen 6-network ensemble; NicheTrans and MLP = 3-seed ensembles; others single models
XMAP = {'txcellnet': 'txcellnet', 'nichetrans': 'nichetrans_off_e3', 'dgat': 'dgat_pm', 'spamosaic': 'spamosaic_pm', 'totalvi': 'totalvi_pm',
        'mlp': 'mlp_pm_e3', 'ridge': 'ridge_pm', 'knn': 'knn_pm', 'celltype': 'celltype_pm', 'own_mRNA': 'own_mRNA'}
# single models for the within-versus-across comparison
XSINGLE = dict(XMAP, txcellnet=TX_SINGLE, nichetrans='nichetrans_off', mlp='mlp_pm')

def build(mode, cases):
    if mode == 'within':
        kid = [c for c in cases if c.startswith('kidney')]; oth = [c for c in cases if not c.startswith('kidney')]
        T_ = pd.DataFrame({k: pd.concat([pick(CROSS, KMAP[k], kid), pick(WITHIN, WMAP[k], oth)]) for k, *_ in METHODS})
    else:
        T_ = pd.DataFrame({k: pick(CROSS, XMAP[k], cases) for k, *_ in METHODS})
    return T_.reindex(cases)


T = build(MODE, ALL)
ms = [m for m in METHODS if T[m[0]].notna().any()]
G = pd.DataFrame({short: T.reindex(cs).mean() for _, short, cs in GROUPS}); G['All'] = T.mean(); G = G.loc[[m[0] for m in ms]]
T.to_csv(f'{OUT}/fig_{MODE}_per_dataset.csv'); G.to_csv(f'{OUT}/fig_{MODE}_groups.csv')
print(G.round(3).to_string())
print('wins of TxCellNet:', {k: f'{int(((T.txcellnet - T[k]) > 0).sum())}/{int((T.txcellnet - T[k]).notna().sum())}' for k, *_ in ms if k != 'txcellnet'})


# ---------------- numbers for the text (JSON) and the supplementary table (LaTeX) ----------------
keys = [m[0] for m in ms]


def stats(T_):
    have = T_[keys]
    best = have.max(1)
    out = dict(n=int(T_.txcellnet.notna().sum()), mean=have.mean().round(4).to_dict(),
               wins={k: [int(((T_.txcellnet - T_[k]) > 0).sum()), int((T_.txcellnet - T_[k]).notna().sum())] for k in keys if k != 'txcellnet'},
               best_count=int((T_.txcellnet >= best - 1e-12).sum()),
               rank=have.mean().sort_values(ascending=False).round(4).to_dict())
    return out


S = dict(mode=MODE, all=stats(T), groups=G.round(4).to_dict())
if MODE == 'cross':
    S['test'] = stats(T.reindex(TEST)); S['fresh'] = stats(T.reindex(FRESH))
    DEV = BREAST((1,)) + ['xen_lung_v1', 'xen_pancreas_v1', 'crlm_tumor1', 'crlm_tumor3_A1', 'duchini_core01_hcc']
    S['dev'] = stats(build('cross', DEV)); S['kidney_heldout'] = build('cross', ['kidney_rcc', 'kidney_ccrcc']).mean().round(4).to_dict()
    tc = {}
    for k in keys:
        if k == 'own_mRNA': continue
        wv = pick(WITHIN, WMAP[k], ALL); xv = pick(CROSS, XSINGLE[k], ALL); ok = wv.notna() & xv.notna()
        if ok.sum(): tc[k] = dict(n=int(ok.sum()), within=round(float(wv[ok].mean()), 4), cross_single=round(float(xv[ok].mean()), 4))
    S['transfer_cost'] = tc
else:
    S['kidney'] = T.reindex(['kidney_rcc', 'kidney_ccrcc']).round(4).to_dict()
json.dump(S, open(f'{OUT}/fig_{MODE}_stats.json', 'w'), indent=1)

COLS = [('txcellnet', 'TxCellNet'), ('nichetrans', 'NT'), ('dgat', 'DGAT'), ('spamosaic', 'SpaM.'), ('totalvi', 'totalVI'), ('mlp', 'MLP'),
        ('ridge', 'Ridge'), ('knn', 'kNN'), ('celltype', 'Type'), ('own_mRNA', 'mRNA')]
E3 = {'nichetrans', 'mlp'} if MODE == 'cross' else set()  # three-seed ensembles (second header row)
if MODE == 'cross':
    SPL = [('Kidney, held-out stripes', ['kidney_rcc', 'kidney_ccrcc']),
           ('Development sections', BREAST((1,)) + ['xen_lung_v1', 'xen_pancreas_v1', 'crlm_tumor1', 'crlm_tumor3_A1', 'duchini_core01_hcc']),
           ('Test sections', TEST), ('Fresh datasets', FRESH)]
else:
    SPL = [(g, cs) for g, _, cs in GROUPS]
def NAME(c):
    if c.startswith('kidney'): return {'kidney_rcc': 'Kidney RCC', 'kidney_ccrcc': 'Kidney ccRCC'}[c]
    if c.startswith('xen_breast_S'): return 'Breast ' + SHORT[c].replace('sample ', 'S')
    if c.startswith('crlm'): return 'CRLM ' + SHORT[c]
    n = SHORT.get(c, c); return n[:1].upper() + n[1:]
num = lambda v, d: f'{v:.{d}f}'.replace('-', '$-$')
lines = ['\\begin{tabular}{@{}l' + 'r' * len(COLS) + '@{}}', '\\toprule', 'Section & ' + ' & '.join(n for _, n in COLS) + '\\\\']
if E3: lines.append(' ' + ' '.join(('& ' + ('$\\times$3' if k in E3 else '')).strip() for k, _ in COLS) + ' \\\\')
lines.append('\\midrule')
for title, cs in SPL:
    TT = build(MODE, cs)
    lines.append(f'\\multicolumn{{{len(COLS) + 1}}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\')
    for c in cs:
        vals = [TT.loc[c, k] for k, _ in COLS]; fin = [v for v in vals if v == v]
        if not fin: continue
        b = max(fin)
        lines.append(NAME(c) + ' & ' + ' & '.join('--' if v != v else (f'\\textbf{{{num(v, 2)}}}' if v == b else num(v, 2)) for v in vals) + '\\\\')
    lines.append('\\textit{Mean} & ' + ' & '.join('--' if TT[k].isna().all() else num(TT[k].mean(), 3) for k, _ in COLS) + '\\\\')
    lines.append('\\midrule')
lines[-1] = '\\bottomrule'; lines.append('\\end{tabular}')
open(f'{OUT}/table_{MODE}.tex', 'w').write('\n'.join(lines) + '\n')
print(json.dumps(S['all'], indent=None)[:1500]); print('saved', OUT)
