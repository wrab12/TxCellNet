"""Semi-synthetic benchmark (sim_make.py): the published methods DGAT, SpaMosaic and totalVI, evaluated exactly as the
cell-level baselines of sim_eval.py: the same perturbation (same random generator sequence), the same per-cell targets
(mean of the noisy synthetic image over the input segmentation), the same training cells (training stripes of both
sections), per-cell predictions painted uniformly into the input cell masks, and the same metrics against the noiseless
ground truth on the held-out stripes. As for the other cell-level baselines, the methods are retrained on the perturbed
inputs. The methods run with the code of dgat_bench.py, spamosaic_bench.py and totalvi_bench.py on the genes measured in
both sections; every cell is predicted from its RNA.
modes:
  prep PERT           (main env) per-cell counts, targets, training mask and centroids -> data/sim/sota_cache/PERT_*
  run METHOD PERT     (pyg env: dgat, spamosaic; scvi env: totalvi) train and predict -> results/sim/sota/pred_METHOD_PERT.npz
                      (SPAM_DEVICE=cpu: SpaMosaic on the CPU - its documented device option - when the graph of a
                      perturbation does not fit in GPU memory)
  eval PERT           (main env) metrics of all available predictions -> results/sim/sota/sim_rows_PERT.csv
                      (NicheTrans: pred_nichetrans_PERT.npz of nichetrans_official.py sim PERT, official tutorial training)
PERT in {none, thin50, thin25, dilate2, merge20}
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp

SIM = 'data/sim'; SECT = ['rcc', 'ccrcc']; CACHE = f'{SIM}/sota_cache'; OUT = 'results/sim/sota'
MODE, ARGS = sys.argv[1], sys.argv[2:]
CH = json.load(open(f'{SIM}/rcc/meta.json'))['prots']
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def stripes(s, W, um):
    edges = np.array(json.load(open(f'results/cell_baseline/{s}_stripes.json'))['x_edges_um']) / um
    return np.digitize(np.arange(W), edges) % 2 == 1  # test columns


def perturb_labels(lab, PERT, rng):  # identical to sim_eval.py
    if PERT == 'dilate2':
        import cv2
        out = lab.copy()
        for _ in range(2):
            d = cv2.dilate(out.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
            out = np.where(out == 0, d, out)
        return out
    if PERT == 'merge20':
        a, b = lab[:, :-1].ravel(), lab[:, 1:].ravel(); k = (a != b) & (a > 0) & (b > 0)
        pairs = np.unique(np.stack([a[k], b[k]], 1), axis=0)
        sel = pairs[rng.random(len(pairs)) < 0.2]
        mp = np.arange(lab.max() + 1)
        for x, y in sel: mp[y] = mp[x]
        return mp[lab]
    return lab


def build(PERT):
    """sections with the perturbed input segmentation and molecules, consuming the random generator as sim_eval.py."""
    rng = np.random.default_rng(0); D = {}
    for s in SECT:
        sd = f'{SIM}/{s}'; meta = json.load(open(f'{sd}/meta.json'))
        lab = np.load(f'{sd}/cells.npy'); H, W = lab.shape; test_col = stripes(s, W, meta['um_per_px'])
        lab_in = perturb_labels(lab, PERT, rng)
        T = np.load(f'{sd}/tx.npz'); gy, gx, gid = T['gy'].astype(np.int64), T['gx'].astype(np.int64), T['gid'].astype(np.int64)
        if PERT.startswith('thin'):
            keep = rng.random(len(gid)) < int(PERT[4:]) / 100; gy, gx, gid = gy[keep], gx[keep], gid[keep]
        D[s] = dict(meta=meta, lab=lab, lab_in=lab_in, test_col=test_col, gy=gy, gx=gx, gid=gid, H=H, W=W)
    return D


if MODE == 'prep':
    PERT = ARGS[0]; D = build(PERT)
    both = sorted(set(D['rcc']['meta']['genes']) & set(D['ccrcc']['meta']['genes']))
    for s in SECT:
        d = D[s]; lab_in = d['lab_in']; n = int(lab_in.max()) + 1; genes = d['meta']['genes']; pos = {g: i for i, g in enumerate(genes)}
        L = lab_in[d['gy'], d['gx']]
        X = sp.csr_matrix((np.ones(len(L), np.float32), (L, d['gid'])), shape=(n, len(genes)))[:, [pos[g] for g in both]].tocsr()
        yy, xx = np.nonzero(lab_in); Ll = lab_in[yy, xx]; cnt = np.bincount(Ll, minlength=n).astype(np.float64)
        cen = np.stack([np.bincount(Ll, weights=xx, minlength=n), np.bincount(Ll, weights=yy, minlength=n)], 1) / np.maximum(cnt, 1)[:, None]
        Y = np.load(f'{SIM}/{s}/Y.npy', mmap_mode='r')
        tgt = np.stack([np.bincount(Ll, weights=np.asarray(Y[j], np.float32)[yy, xx].astype(np.float64), minlength=n) / np.maximum(cnt, 1)
                        for j in range(len(CH))], 1)
        train = (cnt > 0) & ~d['test_col'][np.clip(cen[:, 0].astype(int), 0, d['W'] - 1)]; train[0] = False
        node = cnt > 0; node[0] = False
        sp.save_npz(f'{CACHE}/{PERT}_{s}_X.npz', X)
        np.savez(f'{CACHE}/{PERT}_{s}.npz', tgt=tgt.astype(np.float32), train=train, node=node, xy=(cen * d['meta']['um_per_px']).astype(np.float32))
        log(PERT, s, 'cells', n, 'nodes', int(node.sum()), 'train', int(train.sum()), 'genes', len(both))
    json.dump(both, open(f'{CACHE}/{PERT}_genes.json', 'w'))

elif MODE == 'run':
    METHOD, PERT = ARGS
    genes = json.load(open(f'{CACHE}/{PERT}_genes.json')); names = [f'p{j}' for j in range(len(CH))]
    S = {s: dict(X=sp.load_npz(f'{CACHE}/{PERT}_{s}_X.npz').tocsr(), **dict(np.load(f'{CACHE}/{PERT}_{s}.npz'))) for s in SECT}
    for s in SECT:
        tr = S[s]['train']; mu_, sd_ = S[s]['tgt'][tr].mean(0), S[s]['tgt'][tr].std(0) + 1e-6
        S[s]['z'] = (S[s]['tgt'] - mu_) / sd_; S[s]['raw'] = np.clip(S[s]['tgt'], 0, None)  # intensities are non-negative
    pred = {}
    if METHOD == 'dgat':
        import dgat_bench as G, torch
        nodes = {s: np.flatnonzero(S[s]['node']) for s in SECT}
        Xtr = sp.vstack([S[s]['X'][S[s]['train']] for s in SECT]).tocsr()
        keep = G.dgat_genes(Xtr, genes, np.arange(Xtr.shape[0]), set())  # DGAT gene filter on the training cells
        ki = [genes.index(g) for g in keep]
        Xr = {s: G.rna_scale(S[s]['X'][S[s]['node']][:, ki]) for s in SECT}  # per-sample normalisation on the kept genes
        ki = list(range(len(keep)))
        graphs = []
        for s in SECT:
            pos = np.full(len(S[s]['node']), -1); pos[nodes[s]] = np.arange(len(nodes[s]))
            trn = pos[np.flatnonzero(S[s]['train'])]; Pc = G.clr0(S[s]['raw'][S[s]['train']])
            Xs = Xr[s][:, ki]; xy = S[s]['xy'][nodes[s]]
            graphs += [G.tile_graph(Xs[trn[t]], xy[trn[t]], Pc[t]) for t in G.tiles(xy[trn], G.TILE_TRAIN)]
        log('DGAT graphs', len(graphs), 'genes', len(keep))
        enc_r, dec_p = G.train_dgat(graphs, len(keep), len(CH))
        for s in SECT:
            p = np.full((len(S[s]['node']), len(CH)), np.nan, np.float32)
            p[nodes[s]] = G.predict(enc_r, dec_p, Xr[s][:, ki], S[s]['xy'][nodes[s]], len(CH)); pred[s] = p
    elif METHOD == 'spamosaic':
        import spamosaic_bench as Sm, scanpy as sc, torch
        from spamosaic.framework import SpaMosaic
        from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess
        rna, adt, qidx = [], [], {}
        for s in SECT:  # bridges: training cells (RNA + protein), at most N_REF each
            t = Sm.sub(np.flatnonzero(S[s]['train']), Sm.N_REF); idx = [f'{s}_tr_{i}' for i in range(len(t))]
            a = sc.AnnData(S[s]['X'][t].astype(np.float32), obs=pd.DataFrame({'src': f'{s}_train'}, index=idx)); a.var_names = genes
            a.obsm['spatial'] = S[s]['xy'][t]; a.layers['counts'] = a.X
            b = sc.AnnData(S[s]['raw'][t].astype(np.float32), obs=pd.DataFrame({'src': f'{s}_train'}, index=idx)); b.var_names = names
            b.obsm['spatial'] = S[s]['xy'][t]; b.layers['counts'] = S[s]['z'][t].astype(np.float32); rna.append(a); adt.append(b)
        for s in SECT:  # queries: every cell of the section (RNA only)
            q = np.flatnonzero(S[s]['node']); qidx[s] = q
            a = sc.AnnData(S[s]['X'][q].astype(np.float32), obs=pd.DataFrame({'src': f'{s}_all'}, index=[f'{s}_q_{i}' for i in range(len(q))]))
            a.var_names = genes; a.obsm['spatial'] = S[s]['xy'][q]; a.layers['counts'] = a.X; rna.append(a); adt.append(None)
        RNA_preprocess(rna, batch_corr=True, favor='scanpy', n_hvg=5000, batch_key='src', key='dimred_bc')
        ADT_preprocess(adt, batch_corr=True, batch_key='src', key='dimred_bc')
        D_ = {'rna': rna, 'adt': adt}
        for seed in (1234, 1235, 1236):
            model = SpaMosaic(modBatch_dict=D_, input_key='dimred_bc', batch_key='src', intra_knns=10, inter_knn_base=10, w_g=0.8, seed=seed, device=os.environ.get('SPAM_DEVICE', 'cuda:0'))
            model.train(net='wlgcn', lr=0.01, T=0.01, n_epochs=100)
            if np.isfinite(model.loss_cl).all() and np.isfinite(model.loss_rec).all(): break
            log('SpaMosaic training diverged with seed', seed)
        model.infer_emb(D_, emb_key='emb', final_latent_key='merged_emb')
        imp = model.impute(D_, emb_key='emb', layer_key='counts', imp_knn=10)['adt']
        for k, s in enumerate(SECT):
            p = np.full((len(S[s]['node']), len(CH)), np.nan, np.float32); p[qidx[s]] = np.asarray(imp[2 + k], np.float32); pred[s] = p
    elif METHOD == 'totalvi':
        import totalvi_bench as Tv, anndata as ad, scvi
        sc_ = 500.0 / np.maximum(np.percentile(np.vstack([S[s]['raw'][S[s]['train']] for s in SECT]), 99, axis=0), 1e-6)
        Xs, Ps, bs = [], [], []
        for s in SECT:
            t = Tv.sub(np.flatnonzero(S[s]['train']), Tv.N_REF); Xs.append(S[s]['X'][t]); Ps.append(np.round(S[s]['raw'][t] * sc_)); bs += [f'{s}_train'] * len(t)
        for s in SECT:
            q = Tv.sub(np.flatnonzero(S[s]['node']), Tv.N_QUERY // 2); Xs.append(S[s]['X'][q]); Ps.append(np.zeros((len(q), len(CH)))); bs += [f'{s}_all'] * len(q)
        Xa = sp.vstack(Xs).tocsr(); lib = np.asarray(Xa.sum(1)).ravel() > 0; cats = [f'{s}_train' for s in SECT] + [f'{s}_all' for s in SECT]
        A = ad.AnnData(Xa[lib].toarray().astype(np.float32), obs=pd.DataFrame({'batch': pd.Categorical(np.array(bs)[lib], categories=cats)},
                                                                            index=[f'c{i}' for i in range(int(lib.sum()))]))
        A.var_names = genes; A.obsm['protein_expression'] = pd.DataFrame(np.vstack(Ps)[lib].astype(np.float32), index=A.obs_names, columns=names)
        scvi.model.TOTALVI.setup_anndata(A, batch_key='batch', protein_expression_obsm_key='protein_expression')
        model = scvi.model.TOTALVI(A, latent_distribution='normal', n_layers_decoder=2)
        model.train(max_epochs=min(round(20000 / A.n_obs * 400), 400), enable_progress_bar=False)
        for s in SECT:
            q = np.flatnonzero(S[s]['node'])
            B = ad.AnnData(S[s]['X'][q].tocsr().astype(np.float32), obs=pd.DataFrame({'batch': pd.Categorical([f'{s}_all'] * len(q), categories=cats)},
                                                                                   index=[f'q{i}' for i in range(len(q))]))
            B.var_names = genes; B.obsm['protein_expression'] = pd.DataFrame(np.zeros((len(q), len(CH)), np.float32), index=B.obs_names, columns=names)
            _, prot = model.get_normalized_expression(B, n_samples=25, return_mean=True, transform_batch=[f'{x}_train' for x in SECT],
                                                      include_protein_background=True, sample_protein_mixing=False, gene_list=[genes[0]], batch_size=1024)
            p = np.full((len(S[s]['node']), len(CH)), np.nan, np.float32); p[q] = np.log1p(np.asarray(prot, np.float32)); pred[s] = p
    np.savez(f'{OUT}/pred_{METHOD}_{PERT}.npz', **pred); log(METHOD, PERT, 'saved')

elif MODE == 'eval':
    from scipy.stats import spearmanr
    PERT = ARGS[0]; D = build(PERT); rows = []
    for METHOD, NAME in (('nichetrans', 'NicheTrans'), ('dgat', 'DGAT'), ('spamosaic', 'SpaMosaic'), ('totalvi', 'totalVI')):
        f = f'{OUT}/pred_{METHOD}_{PERT}.npz'
        if not os.path.exists(f): continue
        P = np.load(f)
        for s in SECT:
            d = D[s]; lab = d['lab']; tc = d['test_col'][None, :] & (lab > 0)
            cp = P[s].astype(np.float32); cp = np.where(np.isfinite(cp), cp, np.nanmean(cp, 0))  # cells without prediction (none scored)
            n = int(lab.max()) + 1; cnt = np.bincount(lab[tc], minlength=n).astype(np.float64); test_cells = np.flatnonzero(cnt >= 10)
            Yc = np.load(f'{SIM}/{s}/Yclean.npy', mmap_mode='r'); lin = d['lab_in']; l = lab[tc]
            for j, c in enumerate(CH):  # identical to sim_eval.evaluate with painted per-cell predictions
                p = cp[:, j][lin][tc].astype(np.float64); y = np.asarray(Yc[j], np.float32)[tc].astype(np.float64)
                pm = np.bincount(l, weights=p, minlength=n) / np.maximum(cnt, 1); ym = np.bincount(l, weights=y, minlength=n) / np.maximum(cnt, 1)
                pr, yr = p - pm[l], y - ym[l]
                within = float(np.corrcoef(pr, yr)[0, 1]) if pr.std() > 1e-9 else 0.0
                rows.append(dict(method=NAME, section=s, channel=c, perturb=PERT, cell_spearman=float(spearmanr(pm[test_cells], ym[test_cells])[0]),
                                 pixel_pearson=float(np.corrcoef(p, y)[0, 1]), within_cell=within))
        log(NAME, PERT, 'evaluated')
    df = pd.DataFrame(rows); df.to_csv(f'{OUT}/sim_rows_{PERT}.csv', index=False)
    print(df.groupby('method')[['cell_spearman', 'pixel_pearson', 'within_cell']].mean().round(3).to_string())
