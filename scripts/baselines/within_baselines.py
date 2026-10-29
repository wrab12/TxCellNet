"""Within-dataset cell-level baselines on one prepared case: trained on the cells of the training stripes (same split as
train_case.py), all panel genes, targets = the case's measured per-cell values; scored on the held-out stripes.
Methods: ridge, MLP, kNN reference mapping (PCA-50), cell-type lookup (30 k-means clusters), NicheTrans (official model
class, 12 spatial neighbours, short training; not used in the paper, whose NicheTrans follows the official tutorial:
nichetrans_official.py), cognate mRNA. Features: log1p(normalize_total 1e3) over the panel genes detected in >= 0.1% of
training cells, at most the 2,000 most variable (5K panels); models are fitted on at most 200,000 training cells.
usage: within_baselines.py CASE [CASE ...]   -> results/bench/within/CASE__baselines.csv
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.environ.get('NICHETRANS_DIR', 'external/NicheTrans'))
from model.nicheTrans import NicheTrans

dev = 'cuda'; os.makedirs('results/bench/within', exist_ok=True)


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def batched(fn, n, bs=8192):
    with torch.no_grad(): return np.vstack([fn(torch.arange(i, min(i + bs, n), device=dev)) for i in range(0, n, bs)])


class MLP(nn.Module):
    def __init__(s, d, o):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(.2), nn.Linear(512, 256), nn.GELU(), nn.Dropout(.2), nn.Linear(256, o))

    def forward(s, x): return s.f(x)


def fit_torch(model, X, Y, tr, epochs, lr, wd, nb=None):
    """X, Y numpy (all cells, on CPU); batches are moved to the GPU; nb: neighbour indices (NicheTrans)."""
    torch.manual_seed(1); opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=wd); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train(); perm = np.random.default_rng(ep).permutation(tr)
        for i in range(0, max(len(perm) - 255, 1), 1024):
            b = perm[i:i + 1024]; xb = torch.from_numpy(X[b]).to(dev); yb = torch.from_numpy(Y[b]).to(dev)
            out = model(xb, torch.from_numpy(X[nb[b]]).to(dev)) if nb is not None else model(xb)
            loss = nn.functional.mse_loss(out, yb); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval(); return model


def predict(fn, n, bs=8192):
    with torch.no_grad(): return np.vstack([fn(np.arange(i, min(i + bs, n))) for i in range(0, n, bs)])


for case in sys.argv[1:]:
    cd = f'data/cases/{case}'; meta = json.load(open(f'{cd}/meta.json')); yj = json.load(open(f'{cd}/Y.json'))
    cells = pd.read_parquet(f'{cd}/cells.parquet'); lab = np.load(f'{cd}/lab.npy'); W = lab.shape[1]; n = int(lab.max()) + 1
    Xc = sp.load_npz(f'{cd}/counts.npz').tocsr(); cg = json.load(open(f'{cd}/counts_genes.json'))
    edges = np.array(json.load(open(f'{cd}/split.json'))['x_edges_px'])
    ys_, xs_ = np.nonzero(lab); lv = lab[ys_, xs_]; xmean = np.bincount(lv, weights=xs_, minlength=n) / np.maximum(np.bincount(lv, minlength=n), 1)
    tot = np.asarray(Xc.sum(1)).ravel()
    ok = (cells['area_px'].values >= 10) & (tot >= 1)
    if 'eval_mask' in meta: ok &= cells[meta['eval_mask']].values.astype(bool)
    stripe_test = np.digitize(xmean[cells['label'].values], edges) % 2 == 1
    tr, te = np.flatnonzero(ok & ~stripe_test), np.flatnonzero(ok & stripe_test)
    keys = [k for k in meta['markers'] if k in yj['markers']]
    T = []
    for k in keys:
        mk = meta['markers'][k]; t = cells[mk['truth']].values.astype(np.float64)
        T.append(np.log1p(np.clip(t - mk.get('offset', 0), 0, None)) if mk.get('log', True) else t)
    T = np.stack(T, 1).astype(np.float32); mu, sd = T[tr].mean(0), T[tr].std(0) + 1e-6; Tz = (T - mu) / sd
    det = np.asarray((Xc[tr] > 0).mean(0)).ravel(); use = np.flatnonzero(det >= 1e-3)
    A = Xc[:, use].astype(np.float32); A = (sp.diags(1e3 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A).tocsr(); A.data = np.log1p(A.data)
    if len(use) > 2000:  # most variable genes on the training cells
        At = A[tr]; m1 = np.asarray(At.mean(0)).ravel(); m2 = np.asarray(At.multiply(At).mean(0)).ravel(); top = np.argsort(m2 - m1 ** 2)[::-1][:2000]
        A, use = A[:, np.sort(top)], use[np.sort(top)]
    F_ = A.toarray().astype(np.float32)
    tr = tr if len(tr) <= 200_000 else np.sort(np.random.default_rng(0).choice(tr, 200_000, replace=False))
    log(case, 'train cells', len(tr), 'test cells', len(te), 'genes', len(use), 'markers', keys)
    fm, fs = F_[tr].mean(0), F_[tr].std(0) + 1e-6; Z = ((F_ - fm) / fs).astype(np.float32)
    preds = {}
    Zt = Z[tr].astype(np.float64)
    Wr = np.linalg.solve(Zt.T @ Zt + 10.0 * np.eye(Zt.shape[1]), Zt.T @ Tz[tr]); preds['ridge'] = Z @ Wr
    mlp = fit_torch(MLP(Z.shape[1], len(keys)).to(dev), Z, Tz, tr, 12, 1e-3, 1e-4)
    preds['mlp'] = predict(lambda b: mlp(torch.from_numpy(Z[b]).to(dev)).cpu().numpy(), len(Z))
    sub = np.random.default_rng(0).choice(len(tr), min(len(tr), 100_000), replace=False)
    _, _, Vt = np.linalg.svd(Zt[sub], full_matrices=False); pca = Vt[:50].T.astype(np.float32); Ep = (Z @ pca).astype(np.float32)
    km = MiniBatchKMeans(30, random_state=0, batch_size=8192, n_init=3).fit(Ep[tr]); cmean = np.stack([Tz[tr][km.labels_ == c].mean(0) for c in range(30)])
    preds['celltype'] = cmean[km.predict(Ep)]
    Eg, R_, RP = torch.from_numpy(Ep).to(dev), torch.from_numpy(Ep[tr]).to(dev), torch.from_numpy(Tz[tr]).to(dev)
    preds['knn'] = batched(lambda b: RP[torch.cdist(Eg[b], R_).topk(50, largest=False).indices].mean(1).cpu().numpy(), len(Z), bs=256)
    nb = NearestNeighbors(n_neighbors=13).fit(cells[['x_um', 'y_um']].values).kneighbors(cells[['x_um', 'y_um']].values, return_distance=False)[:, 1:]
    nt = fit_torch(NicheTrans(source_length=F_.shape[1], target_length=len(keys), noise_rate=0.2, dropout_rate=0.2).to(dev), F_, Tz, tr, 8, 1e-3, 5e-4, nb=nb)
    preds['nichetrans'] = predict(lambda b: nt(torch.from_numpy(F_[b]).to(dev), torch.from_numpy(F_[nb[b]]).to(dev)).cpu().numpy(), len(F_), bs=4096)
    gi = {g: i for i, g in enumerate(cg)}
    rows = []
    for j, k in enumerate(keys):
        for mn, pc in preds.items():
            rows.append(dict(case=case, marker=k, method=f'within:{mn}', spearman=float(spearmanr(pc[te, j], T[te, j])[0]), n=len(te)))
        idx = [gi[g] for g in meta['markers'][k].get('own', []) if g in gi]
        if idx:
            own = np.log1p(np.asarray(Xc[:, idx].sum(1)).ravel() / np.maximum(tot, 1) * np.median(tot[tot > 0]))
            rows.append(dict(case=case, marker=k, method='within:own_mRNA', spearman=float(spearmanr(own[te], T[te, j])[0]), n=len(te)))
    df = pd.DataFrame(rows); df.to_csv(f'results/bench/within/{case}__baselines.csv', index=False)
    os.makedirs(f'{cd}/runs/within_baselines', exist_ok=True)
    for mn, pc in preds.items(): np.save(f'{cd}/runs/within_baselines/{mn}.npy', (pc * sd + mu).astype(np.float32))
    log(case, '\n' + df.groupby('method').spearman.median().round(3).to_string())
