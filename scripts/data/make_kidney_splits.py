"""Spatial split of the two kidney training sections and the vendor per-cell protein values of their held-out cells.

Each section is cut into 8 vertical stripes with equal numbers of cells: the 7 edges are the 1/8, ..., 7/8 quantiles of the
vendor centroid x (x_centroid) of the cells with >= 10 gene counts; odd stripes (0-based) are held out, even stripes train.
Outputs (read by training and evaluation):
  results/cell_baseline/<s>_stripes.json        {"x_edges_um": [...]}
  results/cell_baseline/<s>_true_test.parquet   log1p of the vendor per-cell protein values of the held-out cells
usage: make_kidney_splits.py      (reads data/xenium_protein_{rcc,ccrcc}/outs)
"""
import json, os
import h5py, numpy as np, pandas as pd, scipy.sparse as sp

OUT = 'results/cell_baseline'


def load(ds):
    root = f'data/xenium_protein_{ds}/outs'
    with h5py.File(f'{root}/cell_feature_matrix.h5', 'r') as f:
        m = f['matrix']
        X = sp.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=tuple(m['shape'][:])).T.tocsr()
        name = np.array([x.decode() for x in m['features']['name'][:]])
        ftype = np.array([x.decode() for x in m['features']['feature_type'][:]])
        bc = np.array([x.decode() for x in m['barcodes'][:]])
    cells = pd.read_parquet(f'{root}/cells.parquet').set_index('cell_id').loc[bc]
    g = ftype == 'Gene Expression'
    p = ftype == 'Protein Expression'
    G = X[:, g].astype(np.float32)
    P = np.asarray(X[:, p].todense(), dtype=np.float32)
    keep = (np.asarray(G.sum(1)).ravel() >= 10)  # drop near-empty cells
    return dict(P=P[keep], prots=name[p], bc=bc[keep], xy=cells[['x_centroid', 'y_centroid']].values[keep], n_all=len(bc))


def stripe_edges(xy, n=8):
    return np.quantile(xy[:, 0], np.linspace(0, 1, n + 1)[1:-1])


def stripes(x, edges):
    return np.digitize(x, edges) % 2 == 1  # True = test


def main():
    os.makedirs(OUT, exist_ok=True)
    for ds in ['rcc', 'ccrcc']:
        s = load(ds)
        Y = np.log1p(np.clip(s['P'], 0, None))
        edges = stripe_edges(s['xy']); te = stripes(s['xy'][:, 0], edges)
        json.dump(dict(x_edges_um=edges.tolist()), open(f'{OUT}/{ds}_stripes.json', 'w'))
        idx = pd.Index(s['bc'][te], name='cell_id')
        pd.DataFrame(Y[te], columns=s['prots'], index=idx).to_parquet(f'{OUT}/{ds}_true_test.parquet')
        print(ds, 'cells', len(s['bc']), '/', s['n_all'], 'held out', int(te.sum()), 'proteins', len(s['prots']), flush=True)


if __name__ == '__main__':
    main()
