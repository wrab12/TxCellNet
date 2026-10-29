"""Fixed gene vectors for the PCA ablation: each gene of the kidney union vocabulary (495) gets its loadings on the top
64 principal components of standardised log1p(normalize_total 1e3) expression of the kidney train-stripe cells (both
sections, at most 150,000 random cells each; a gene measured in one section only is computed from that section's cells),
multiplied by the singular values / sqrt(N) and scaled to the learned embeddings' initial SD (0.1).
-> results/bench/pca_gene_emb.npy (495, 64)
"""
import json
import numpy as np, scipy.sparse as sp, h5py, pandas as pd

R, C = 'data/xenium_protein_rcc/pilot_L2', 'data/xenium_protein_ccrcc/pilot_L2'
UNION = sorted(set(json.load(open(f'{R}/meta.json'))['genes']) | set(json.load(open(f'{C}/meta.json'))['genes']))
up = {g: i for i, g in enumerate(UNION)}
Z, M = [], []
for s in ['rcc', 'ccrcc']:
    with h5py.File(f'data/xenium_protein_{s}/outs/cell_feature_matrix.h5', 'r') as f:
        m = f['matrix']; bc = np.array([b.decode() for b in m['barcodes'][:]])
        ft = np.array([t.decode() for t in m['features']['feature_type'][:]]); nm = np.array([t.decode() for t in m['features']['name'][:]])
        X = sp.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=tuple(m['shape'][:])).T.tocsr()
    G = X[:, ft == 'Gene Expression']; gn = list(nm[ft == 'Gene Expression'])
    xy = pd.read_parquet(f'data/xenium_protein_{s}/outs/cells.parquet').set_index('cell_id').reindex(bc)[['x_centroid']].values[:, 0]
    edges = np.array(json.load(open(f'results/cell_baseline/{s}_stripes.json'))['x_edges_um'])
    tr = (np.digitize(xy, edges) % 2 == 0) & (np.asarray(G.sum(1)).ravel() > 0)
    G = G[tr].astype(np.float32); tot = np.asarray(G.sum(1)).ravel()
    L = np.log1p((sp.diags(1e3 / tot) @ G).toarray())
    F_ = np.zeros((L.shape[0], len(UNION)), np.float32); mk = np.zeros(len(UNION), bool)
    cols = [up[g] for g in gn if g in up]; F_[:, cols] = L[:, [i for i, g in enumerate(gn) if g in up]]; mk[cols] = True
    mu, sd = F_.mean(0), F_.std(0) + 1e-6; F_ = (F_ - mu) / sd; F_[:, ~mk] = 0
    Z.append(F_[np.random.default_rng(0).choice(len(F_), min(len(F_), 150_000), replace=False)])
Zc = np.vstack(Z)
U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
emb = Vt[:64].T * S[:64] / np.sqrt(len(Zc))
emb = (emb / emb.std() * 0.1).astype(np.float32)
np.save('results/bench/pca_gene_emb.npy', emb)
print('saved', emb.shape, 'var explained by 64 PCs', round(float((S[:64] ** 2).sum() / (S ** 2).sum()), 3))
