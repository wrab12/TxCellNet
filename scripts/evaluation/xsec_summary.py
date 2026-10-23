"""Three evaluation tiers on identical test cells.
For every test section B of a multi-section dataset, the held-out stripes of B (the test cells of its within-dataset run)
are scored for
  within          models trained on B's own training stripes (within-dataset tier; saved predictions re-scored)
  cross-section   models trained on the training stripes of another section A of the same dataset (results/bench/xsec)
  cross-dataset   models trained on the kidney training stripes (cross-dataset tier; saved predictions re-scored on
                  these cells, stains combined with the kidney-fitted weights)
TxCellNet: single networks in every tier - clean (within: B's network; cross-section: A's within network; cross-dataset:
fin_clean_s1), randomized (train_case.py PAUG=1 within_rand; fin_rand_s1) and their average (clean + randomized, the
paper's recipe for transfer); the six-network ensemble of the paper is added in the cross-dataset tier.
NicheTrans: seed 1 everywhere; MLP: seed 1.
Kidney (RCC <-> ccRCC): within = the paper's within tier (models trained on both sections) and single-section models on
their own section; cross-section = the single-section model of the other section; no cross-dataset tier.
Metrics: Spearman, Pearson, AUROC (txcellnet.metrics) per marker; median over markers per (B, tier, A, method).
usage: xsec_summary.py OUT_DIR   -> OUT_DIR/{xsec_markers.csv, xsec_datasets.csv, xsec_groups.csv}
"""
import glob, json, os, sys
import numpy as np, pandas as pd

from txcellnet.bench import load_case_cells as load_case, within_split, combine
from txcellnet.metrics import metrics
from txcellnet.utils import log

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
B_ = 'results/bench'
PANELS = json.load(open(f'{B_}/panels.json')); CASE_PANEL = {c: p for p, v in PANELS.items() for c in v['cases']}
NAME = {'txcellnet': 'TxCellNet (clean)', 'txcellnet_rand': 'TxCellNet (randomized)', 'txcellnet_cr': 'TxCellNet (clean + randomized)',
        'txcellnet_ens6': 'TxCellNet (6-network ensemble)', 'nichetrans_off': 'NicheTrans', 'dgat': 'DGAT',
        'spamosaic': 'SpaMosaic', 'totalvi': 'totalVI', 'mlp': 'MLP', 'ridge': 'ridge', 'knn': 'reference mapping',
        'celltype': 'cell-type lookup', 'own_mRNA': 'cognate mRNA'}


def group_of(a, b):
    if a.startswith('xen_breast'): return 'breast, same patient' if a.split('_')[2] == b.split('_')[2] else 'breast, other patient'
    if a.startswith('crlm_tumor3'): return 'CRLM tumour 3, other ROI'
    if a.startswith('crlm'): return 'CRLM, other section'
    if a.startswith('duchini'): return 'COMET, liver <-> tonsil' if 'tonsil' in a + b else 'COMET, other liver core'
    return 'kidney, other patient'


NR = [os.path.basename(f)[:-len('.diverged')].split('__') for f in sorted(glob.glob(f'{B_}/xsec/*.diverged'))]
pd.DataFrame(NR, columns=['train', 'test', 'method']).to_csv(f'{OUT}/xsec_no_result.csv', index=False)  # training diverged
X = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{B_}/xsec/*__*__*.csv'))])
X = X[X.cells == 'heldout']
pairs = X[['train', 'test']].drop_duplicates().values.tolist()
tests = sorted({b for _, b in pairs})
rows = []
for _, r in X.iterrows():
    rows.append(dict(tier='cross-section', train=r.train, test=r.test, group=group_of(r.train, r.test), method=r.method, marker=r.marker,
                     spearman=r.spearman, pearson=r.pearson, auroc=r.auroc))


def score(tier, train, b, method, pred, keys, T, te, group):
    for j, k in enumerate(keys):
        rows.append(dict(tier=tier, train=train, test=b, group=group, method=method, marker=k, **{m: v for m, v in metrics(pred[te, j], T[te, j]).items() if m != 'pos_frac'}))


for b in tests:
    case = load_case(b); tr, te, keys, T, R = within_split(case); grp = '|'.join(sorted({group_of(a, b) for a, bb in pairs if bb == b}))
    cd = f'data/cases/{b}'; yj = json.load(open(f'{cd}/Y.json'))
    # within: B's own models
    tx = {}
    for tag, m in (('within_clean', 'txcellnet'), ('within_rand', 'txcellnet_rand')):
        if not os.path.exists(f'{cd}/runs/{tag}/cellpred.npy'): log('missing', b, tag); continue
        c = json.load(open(f'{cd}/runs/{tag}/cfg.json')); cp = np.load(f'{cd}/runs/{tag}/cellpred.npy')
        tx[m] = cp[:, [c['channels'].index(yj['markers'][k]) for k in keys]].astype(np.float64)
        score('within', b, b, m, tx[m], keys, T, te, grp)
    if len(tx) == 2: score('within', b, b, 'txcellnet_cr', (tx['txcellnet'] + tx['txcellnet_rand']) / 2, keys, T, te, grp)
    for m in ('nichetrans_off', 'dgat', 'spamosaic', 'totalvi', 'mlp', 'ridge', 'knn', 'celltype'):
        f = f'{cd}/runs/within_baselines/{m}.npy'
        if os.path.exists(f): score('within', b, b, m, np.load(f).astype(np.float64), keys, T, te, grp)
        else: log('missing', f)
    own = X[(X.test == b) & (X.method == 'own_mRNA')].groupby('marker')[['spearman', 'pearson', 'auroc']].mean()  # training-free
    for tier in ('within', 'cross-dataset'):
        for k, r in own.iterrows(): rows.append(dict(tier=tier, train=b if tier == 'within' else 'kidney', test=b, group=grp, method='own_mRNA', marker=k, **r.to_dict()))
    # cross-dataset: kidney-trained models (27 kidney proteins, kidney z-units) re-scored on B's held-out stripes
    meta = case['meta']; g = CASE_PANEL[b]

    def kid(pred27):
        return np.stack([combine(pred27, meta['markers'][k]['components'], k) for k in keys], 1)

    ens = [np.load(f'{B_}/predcache/{b}__fin_{r}_s{s}.npy').astype(np.float64) for r in ('clean', 'rand') for s in (1, 2, 3)]
    score('cross-dataset', 'kidney', b, 'txcellnet', kid(ens[0]), keys, T, te, grp)  # fin_clean_s1
    score('cross-dataset', 'kidney', b, 'txcellnet_rand', kid(ens[3]), keys, T, te, grp)  # fin_rand_s1
    score('cross-dataset', 'kidney', b, 'txcellnet_cr', kid((ens[0] + ens[3]) / 2), keys, T, te, grp)
    score('cross-dataset', 'kidney', b, 'txcellnet_ens6', kid(np.mean(ens, 0)), keys, T, te, grp)
    src = {'nichetrans_off': f'{B_}/nt_off_{g}_s1/preds/{b}__nichetrans_off.npy', 'dgat': f'{B_}/dgat_{g}/preds/{b}__dgat_pm.npy',
           'spamosaic': f'{B_}/spam_{b}/preds/{b}__spamosaic_pm.npy', 'totalvi': f'{B_}/tvi_{b}/preds/{b}__totalvi_pm.npy',
           **{m: f'{B_}/cellbase_s1/preds/{b}__{m}_pm.npy' for m in ('mlp', 'ridge', 'knn', 'celltype')}}
    for m, f in src.items():
        if os.path.exists(f): score('cross-dataset', 'kidney', b, m, kid(np.load(f).astype(np.float64)), keys, T, te, grp)
        else: log('missing', f)
    log(b, 'scored')

# kidney: single-section models (cross-section tier) and the paper's within tier
KM = {'nichetrans_off': 'nichetrans_off', 'dgat_pm': 'dgat', 'spamosaic_pm': 'spamosaic', 'totalvi_pm': 'totalvi', 'mlp_pm': 'mlp',
      'ridge_pm': 'ridge', 'knn_pm': 'knn', 'celltype_pm': 'celltype'}
for sec, other in (('rcc', 'ccrcc'), ('ccrcc', 'rcc')):
    fs = glob.glob(f'{B_}/xsec_kidney/*_{sec}/bench_rows.csv') + glob.glob(f'{B_}/xsec_kidney_tx_cr_{sec}/bench_rows.csv')
    if not os.path.exists(f'{B_}/xsec_kidney_tx_cr_{sec}/bench_rows.csv'): fs += glob.glob(f'{B_}/xsec_kidney_tx_{sec}/bench_rows.csv')
    TXK = {f'xs_clean_{sec}': 'txcellnet', f'xs_rand_{sec}': 'txcellnet_rand', f'xs_cr_{sec}': 'txcellnet_cr'}
    for f in fs:
        for _, r in pd.read_csv(f).iterrows():
            m = TXK.get(r.method, KM.get(r.method))
            if m is None: continue
            tier = 'cross-section' if r.case == f'kidney_{other}' else 'within (single section)'
            rows.append(dict(tier=tier, train=f'kidney_{sec}', test=r.case, group='kidney, other patient', method=m, marker=r.marker,
                             spearman=r.spearman, pearson=r.pearson, auroc=r.auroc))
paper = {'txcellnet': ('final', 'fin_clean_s1'), 'txcellnet_rand': ('final', 'fin_rand_s1'), 'txcellnet_ens6': ('final', 'txcellnet'),
         'nichetrans_off': ('nt_off_full_s1', 'nichetrans_off'), 'dgat': ('dgat_full', 'dgat_pm'),
         'mlp': ('cellbase_s1', 'mlp_pm'), 'ridge': ('cellbase_s1', 'ridge_pm'), 'knn': ('cellbase_s1', 'knn_pm'), 'celltype': ('cellbase_s1', 'celltype_pm')}
for b in ('kidney_rcc', 'kidney_ccrcc'):
    src = dict(paper, spamosaic=(f'spam_{b}', 'spamosaic_pm'), totalvi=(f'tvi_{b}', 'totalvi_pm'))
    for m, (d, mn) in src.items():
        R = pd.read_csv(f'{B_}/{d}/bench_rows.csv'); R = R[(R.case == b) & (R.method == mn)]
        for _, r in R.iterrows():
            rows.append(dict(tier='within', train='kidney (both sections)', test=b, group='kidney, other patient', method=m, marker=r.marker,
                             spearman=r.spearman, pearson=r.pearson, auroc=r.auroc))
    R = pd.read_csv(f'{B_}/own/bench_rows.csv'); R = R[(R.case == b) & (R.method == 'own_mRNA')]  # training-free
    for tier in ('within', 'within (single section)', 'cross-section'):
        for _, r in R.iterrows():
            rows.append(dict(tier=tier, train='-', test=b, group='kidney, other patient', method='own_mRNA', marker=r.marker,
                             spearman=r.spearman, pearson=r.pearson, auroc=r.auroc))
    # clean + randomized kidney networks (fin_clean_s1 + fin_rand_s1), scored like run_bench_gen.py on the held-out stripes
    case = load_case(b); cells, meta = case['cells'], case['meta']
    pc = (np.load(f'{B_}/predcache/{b}__fin_clean_s1.npy').astype(np.float64) + np.load(f'{B_}/predcache/{b}__fin_rand_s1.npy').astype(np.float64)) / 2
    for key, mk in meta['markers'].items():
        tt = cells[mk['truth']].values.astype(np.float64)
        if mk.get('log', True): tt = np.log1p(np.clip(tt - mk.get('offset', 0), 0, None))
        r = {k: v for k, v in metrics(combine(pc, mk['components'], key)[case['ok']], tt[case['ok']]).items() if k != 'pos_frac'}
        rows.append(dict(tier='within', train='kidney (both sections)', test=b, group='kidney, other patient', method='txcellnet_cr', marker=key, **r))

D = pd.DataFrame(rows); D['method_name'] = D.method.map(NAME)
D.to_csv(f'{OUT}/xsec_markers.csv', index=False)
P = D.groupby(['group', 'tier', 'train', 'test', 'method', 'method_name'])[['spearman', 'pearson', 'auroc']].median().reset_index()
P.to_csv(f'{OUT}/xsec_datasets.csv', index=False)
# per group and tier: mean over the (A, B) pairs of the group; within / cross-dataset values are per test section B and
# enter once per pair in which B is the test section (same weights as the cross-section tier)
rows = []
pairs_all = pairs + [('kidney_rcc', 'kidney_ccrcc'), ('kidney_ccrcc', 'kidney_rcc')]
for grp in sorted({g for gs in P.group for g in gs.split('|')}):
    S = P[P.group.str.split('|').apply(lambda gs: grp in gs)]
    tests_g = [b for a, b in pairs_all if group_of(a, b) == grp]  # with multiplicity
    for (tier, m), s in S.groupby(['tier', 'method']):
        if tier == 'cross-section':
            s = s[s.group == grp]; v = s[['spearman', 'pearson', 'auroc']].mean(); n = len(s)
        else:
            per_b = s.groupby('test')[['spearman', 'pearson', 'auroc']].mean()
            w = pd.Series(tests_g).value_counts().reindex(per_b.index).fillna(0)
            v = (per_b.mul(w, axis=0).sum() / w.sum()) if w.sum() else per_b.mean(); n = int(w.sum())
        rows.append(dict(group=grp, tier=tier, method=m, method_name=NAME[m], n=n, **v.round(4).to_dict()))
G = pd.DataFrame(rows); G.to_csv(f'{OUT}/xsec_groups.csv', index=False)
order = ['within', 'within (single section)', 'cross-section', 'cross-dataset']
print(G.pivot_table(index=['group', 'method_name'], columns='tier', values='spearman')[[t for t in order if t in set(G.tier)]].round(3).to_string())
log('saved', OUT)
