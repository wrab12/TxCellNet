"""Cross-section tier, cell-level baselines: trained exactly as in within_baselines.py on the training stripes of case A,
then applied to cases B of the same panel group (another section of the same dataset).
Methods: ridge, MLP, kNN reference mapping (PCA-50), cell-type lookup (30 k-means clusters) and the cognate mRNA of B.
Features: log1p(normalize_total 1e3) over the genes of A detected in >= 0.1% of its training cells (at most the 2,000 most
variable for 5K panels), standardised with A's training cells (as the cross-dataset baselines use the kidney statistics);
targets z-scored with A's training cells; models fitted on at most 200,000 training cells.
Scored on B's held-out stripes (the cells of its within run) and on all its evaluated cells (txcellnet.bench.write_xsec).
usage: xsec_baselines.py A B[,B...]   -> results/bench/xsec/A__B__{ridge,mlp,knn,celltype,own_mRNA}.csv
"""
import os, sys
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn
from sklearn.cluster import MiniBatchKMeans

from txcellnet.bench import xsec_train, xsec_test, write_xsec, subset_genes, log

dev = 'cuda'


class MLP(nn.Module):
    def __init__(s, d, o):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(.2), nn.Linear(512, 256), nn.GELU(), nn.Dropout(.2), nn.Linear(256, o))

    def forward(s, x): return s.f(x)


def fit_torch(model, X, Y, tr, epochs, lr, wd):  # within_baselines.fit_torch without the NicheTrans branch
    torch.manual_seed(1); opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=wd); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train(); perm = np.random.default_rng(ep).permutation(tr)
        for i in range(0, max(len(perm) - 255, 1), 1024):
            b = perm[i:i + 1024]; xb = torch.from_numpy(X[b]).to(dev); yb = torch.from_numpy(Y[b]).to(dev)
            loss = nn.functional.mse_loss(model(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval(); return model


def lognorm_dense(X):
    """log1p(normalize_total 1e3) over the columns of X (the genes in use), dense float32 (within_baselines.py)."""
    A = X.astype(np.float32).tocsr(); A = (sp.diags(1e3 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A).tocsr(); A.data = np.log1p(A.data)
    return A.toarray().astype(np.float32)


def batched(fn, n, bs):
    with torch.no_grad(): return np.vstack([fn(torch.arange(i, min(i + bs, n), device=dev)) for i in range(0, n, bs)])


a, bs = sys.argv[1], sys.argv[2].split(',')
A, trA, kA, TA, RA = xsec_train(a); X = A['X']
T = TA.astype(np.float32); mu, sd = T[trA].mean(0), T[trA].std(0) + 1e-6; Tz = (T - mu) / sd
det = np.asarray((X[trA] > 0).mean(0)).ravel(); use = np.flatnonzero(det >= 1e-3)
FA = lognorm_dense(X[:, use]); top = np.arange(len(use))  # normalised over the detected genes (within_baselines.py)
if len(use) > 2000:  # then the 2,000 most variable on the training cells, without renormalising (within_baselines.py)
    m1 = FA[trA].mean(0); m2 = (FA[trA] ** 2).mean(0); top = np.sort(np.argsort(m2 - m1 ** 2)[::-1][:2000]); FA = FA[:, top]
genes = [A['genes'][i] for i in use]  # normalisation genes; features = columns `top`
tr = trA if len(trA) <= 200_000 else np.sort(np.random.default_rng(0).choice(trA, 200_000, replace=False))
fm, fs = FA[tr].mean(0), FA[tr].std(0) + 1e-6; Z = ((FA - fm) / fs).astype(np.float32)
log(a, 'xsec train cells', len(tr), 'genes', len(top), 'of', len(genes), 'targets', kA)
Zt = Z[tr].astype(np.float64)
Wr = np.linalg.solve(Zt.T @ Zt + 10.0 * np.eye(Zt.shape[1]), Zt.T @ Tz[tr])
mlp = fit_torch(MLP(Z.shape[1], len(kA)).to(dev), Z, Tz, tr, 12, 1e-3, 1e-4)
sub = np.random.default_rng(0).choice(len(tr), min(len(tr), 100_000), replace=False)
_, _, Vt = np.linalg.svd(Zt[sub], full_matrices=False); pca = Vt[:50].T.astype(np.float32)
EA = (Z[tr] @ pca).astype(np.float32)
km = MiniBatchKMeans(30, random_state=0, batch_size=8192, n_init=3).fit(EA); cmean = np.stack([Tz[tr][km.labels_ == c].mean(0) for c in range(30)])
R_, RP = torch.from_numpy(EA).to(dev), torch.from_numpy(Tz[tr]).to(dev)
del Zt, FA, Z
for b in bs:
    B, keys, ia, teB, okB, TB = xsec_test(b, kA)
    ZB = ((lognorm_dense(subset_genes(B['X'], B['genes'], genes))[:, top] - fm) / fs).astype(np.float32); EB = (ZB @ pca).astype(np.float32)
    preds = {'ridge': ZB @ Wr}
    Zg = torch.from_numpy(ZB).to(dev)
    preds['mlp'] = batched(lambda i: mlp(Zg[i]).cpu().numpy(), len(ZB), 8192)
    preds['celltype'] = cmean[km.predict(EB)]
    Eg = torch.from_numpy(EB).to(dev)
    preds['knn'] = batched(lambda i: RP[torch.cdist(Eg[i], R_).topk(50, largest=False).indices].mean(1).cpu().numpy(), len(EB), 256)
    for mn, pc in preds.items():
        pred = np.full((len(B['cells']), len(keys)), np.nan, np.float32); pred[okB] = pc[okB][:, ia]
        write_xsec(a, b, mn, keys, pred, TB, teB, okB)
    # cognate mRNA of B (training-free; within_baselines.py own_mRNA)
    gi = {g: i for i, g in enumerate(B['genes'])}; tot = np.asarray(B['X'].sum(1)).ravel()
    own = np.full((len(B['cells']), len(keys)), np.nan, np.float32)
    for j, k in enumerate(keys):
        idx = [gi[g] for g in B['meta']['markers'][k].get('own', []) if g in gi]
        if idx: own[okB, j] = np.log1p(np.asarray(B['X'][:, idx].sum(1)).ravel() / np.maximum(tot, 1) * np.median(tot[tot > 0]))[okB]
    write_xsec(a, b, 'own_mRNA', keys, own, TB, teB, okB)
    del Zg, Eg; torch.cuda.empty_cache()
log('done')
