"""Window arrays of every compared method for the image figures (Fig. 3a, 3b, Fig. 4a, Supplementary Fig. S3), per-cell
predictions painted into the cells.

Windows (WINDOW_DIR): kidney RCC held-out window crops_rcc.npz (crops_kidney.py), CRLM tumour 3 A3 window
crlm_window_crlm_tumor3_A3.npz (crlm_window.py) and the semi-synthetic region sim_region.npz (sim_region.py).
Per-cell predictions (rows = data/cases/<case>/cells.parquet, columns = the 27 kidney proteins), all kidney-trained:
  NicheTrans  official tutorial settings, seed 1 (nt_off_*_s1)          DGAT       dgat_<panel group>
  SpaMosaic   spam_<case>                                              totalVI    tvi_<case>
  baselines   cellbase_s1: MLP, ridge, reference mapping (kNN), cell-type lookup
Semi-synthetic: results/sim/sota/pred_<method>_none.npz (log(x+1) per cell label, unperturbed inputs).
Also collects the per-marker cell-level metrics of the same runs (bench_rows.csv; TxCellNet = 6-network ensemble 'txcellnet').
usage: allmethods_extract.py OUT_DIR WINDOW_DIR
"""
import glob, json, os, sys
import numpy as np, pandas as pd

from txcellnet.bench import PROTS, OWN

OUT, WIN = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
B = 'results/bench'
CELL = {'nichetrans': 'nichetrans_off', 'dgat': 'dgat_pm', 'spamosaic': 'spamosaic_pm', 'totalvi': 'totalvi_pm',
        'mlp': 'mlp_pm', 'ridge': 'ridge_pm', 'knn': 'knn_pm', 'celltype': 'celltype_pm'}


def pred_file(case, m):
    pat = {'nichetrans': f'{B}/nt_off_*_s1/preds/{case}__nichetrans_off.npy', 'dgat': f'{B}/dgat_*/preds/{case}__dgat_pm.npy',
           'spamosaic': f'{B}/spam_{case}/preds/{case}__spamosaic_pm.npy', 'totalvi': f'{B}/tvi_{case}/preds/{case}__totalvi_pm.npy'}.get(
        m, f'{B}/cellbase_s1/preds/{case}__{CELL[m]}.npy')
    fs = sorted(glob.glob(pat)); assert len(fs) == 1, (case, m, fs); return fs[0]


def rows_file(f):  # bench_rows.csv next to the preds folder
    return os.path.join(os.path.dirname(os.path.dirname(f)), 'bench_rows.csv')


def paint(lut, lw):
    return np.where(lw[None] > 0, lut[lw].transpose(2, 0, 1), np.nan).astype(np.float16)


metrics = []

# ---------------- kidney RCC held-out window ----------------
case = 'kidney_rcc'; sd = 'data/xenium_protein_rcc/pilot_L2'
Z = np.load(f'{WIN}/crops_rcc.npz'); lw = Z['lab']; assert [str(p) for p in Z['prots']] == list(PROTS)
ids = pd.read_csv(f'{sd}/cell_ids.csv')['cell_id'].astype(str).values
cells = pd.read_parquet(f'data/cases/{case}/cells.parquet')
rowlab = pd.Series(np.arange(1, len(ids) + 1), index=ids).reindex(cells['cell_id'].astype(str).values).values
ok = ~np.isnan(rowlab)
out = {k: Z[k] for k in ('window', 'prots', 'genes', 'dapi', 'lab', 'ty', 'tx', 'tg', 'measured', 'mean', 'refined')}
for m in CELL:
    f = pred_file(case, m); P = np.load(f).astype(np.float32)
    lut = np.full((len(ids) + 1, len(PROTS)), np.nan, np.float32); lut[rowlab[ok].astype(int)] = P[ok]
    out[m] = paint(lut, lw)
    R = pd.read_csv(rows_file(f)); R = R[(R.case == case) & (R.method == CELL[m])]
    metrics.append(R.assign(method=m)[['case', 'marker', 'method', 'spearman', 'pearson', 'auroc']])
d = np.nanmax(np.abs(out['nichetrans'].astype(np.float32) - Z['nichetrans']))
print('kidney: painted NicheTrans vs stored crops, max |diff| =', d)
R = pd.read_csv(f'{B}/final/bench_rows.csv'); R = R[(R.case == case) & (R.method == 'txcellnet')]
metrics.append(R.assign(method='txcellnet')[['case', 'marker', 'method', 'spearman', 'pearson', 'auroc']])
out['own'] = np.array([json.dumps({p: list(OWN.get(p, [])) for p in PROTS})])
np.savez_compressed(f'{OUT}/kidney_rcc_allm.npz', **out)

# ---------------- CRLM tumour 3 A3 window (IMC) ----------------
case = 'crlm_tumor3_A3'
C = np.load(f'{WIN}/crlm_window_crlm_tumor3_A3.npz'); marks = [str(m) for m in C['marks']]; lw = C['lab']
cells = pd.read_parquet(f'data/cases/{case}/cells.parquet'); L = int(max(lw.max(), cells['label'].max())) + 1
out = {k: C[k] for k in C.files}
for m in CELL:
    f = pred_file(case, m); P = np.load(f).astype(np.float32)
    lut = np.full((L, len(PROTS)), np.nan, np.float32); lut[cells['label'].values] = P
    out[m] = paint(lut[:, [PROTS.index(k) for k in marks]], lw)
    R = pd.read_csv(rows_file(f)); R = R[(R.case == case) & (R.method == CELL[m])]
    metrics.append(R.assign(method=m)[['case', 'marker', 'method', 'spearman', 'pearson', 'auroc']])
d = np.nanmax(np.abs(out['nichetrans'].astype(np.float32) - C['nichetrans']))
print('crlm: painted NicheTrans vs stored window, max |diff| =', d)
R = pd.read_csv(f'{B}/final/bench_rows.csv'); R = R[(R.case == case) & (R.method == 'txcellnet')]
metrics.append(R.assign(method='txcellnet')[['case', 'marker', 'method', 'spearman', 'pearson', 'auroc']])
np.savez_compressed(f'{OUT}/crlm_tumor3_A3_allm.npz', **out)

# ---------------- semi-synthetic region (section rcc) ----------------
S = np.load(f'{WIN}/sim_region.npz'); lr = S['lab_r']
out = {k: S[k] for k in S.files}
for m in ('nichetrans', 'dgat', 'spamosaic', 'totalvi'):
    v = np.expm1(np.load(f'results/sim/sota/pred_{m}_none.npz')['rcc'])
    out[m] = np.where(lr[None] > 0, v[lr].transpose(2, 0, 1), np.nan).astype(np.float32)
out['channels'] = np.array(json.load(open('data/sim/rcc/meta.json'))['prots'])
np.savez_compressed(f'{OUT}/sim_rcc_allm.npz', **out)

D = pd.concat(metrics)
D.to_csv(f'{OUT}/metrics_allm.csv', index=False)
print(D.groupby(['case', 'method']).spearman.median().round(3).unstack('case').to_string())
sim = pd.concat([pd.read_csv(f) for f in ['results/sim/sota/sim_rows_none.csv', 'results/sim/ens/sim_rows_none.csv']])
sim.to_csv(f'{OUT}/sim_rows_none_all.csv', index=False)
print(sim.columns.tolist()); print(sim.method.unique())
print('saved', OUT)
