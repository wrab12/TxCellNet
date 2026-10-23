"""totalVI (Gayoso et al., Nat Methods 2021; scvi-tools 1.5.1) on our benchmark, with the documented recipe for proteins of
RNA-only cells (scvi-tools tutorial 'Integration of CITE-seq and scRNA-seq'): reference cells (RNA + protein) and query
cells (RNA, protein set to zero) are concatenated on the shared genes and trained jointly, batch = sample (a protein that
is zero in a whole batch is treated as missing there); setup_anndata(batch_key, protein_expression_obsm_key),
TOTALVI(latent_distribution='normal', n_layers_decoder=2), train(); the query's proteins are the denoised expression
(get_normalized_expression: transform_batch = the reference batches, include_protein_background=True,
sample_protein_mixing=False, n_samples=25, posterior mean). The query RNA is used for training (transductive).
Genes: the tutorial's sc.pp.highly_variable_genes(adata, batch_key='batch', flavor='seurat_v3', n_top_genes=4000,
subset=True) on the concatenated training data, panels with <= 4,000 genes keep all genes (cross: the panel-matched genes).
max_epochs: scvi's default min(round(20000 / n * 400), 400) (with its early stopping), or MAX_EPOCHS.
Protein inputs must be counts: kidney 10x protein values are integers; other datasets' per-cell intensities (within mode)
are scaled per channel so that the 99th percentile of the training cells is 500 and rounded.
Training cells are subsampled to <= N_REF reference cells per sample (cross: 2 kidney sections) and <= N_QUERY query
cells (cross 100,000 + 100,000 + 100,000; within 200,000 + 100,000); every scored cell is predicted. With scvi's default
number of epochs (inversely proportional to the number of cells) the number of optimisation steps does not depend on it.
Cells without counts on the model's genes cannot be trained on (log library size); they are predicted from the latent space.
modes (as dgat_bench.py):
  cross OUT_DIR GROUP[,GROUP...]   reference = kidney training stripes of both sections, genes = the group's genes;
                                   one model per case (transductive) -> OUT_DIR/bench_rows.csv, OUT_DIR/preds/
  within CASE [CASE ...]           reference = training stripes, query = held-out stripes -> results/bench/within/CASE__totalvi.csv
  xsec A B[,B...]                  cross-section tier: reference = training stripes of case A (as in its within run, protein
                                   scaled with A's training cells), query = every evaluated cell of case B (same panel group);
                                   one model per B -> results/bench/xsec/  (kidney sections: cross mode with KID_TRAIN)
"""
import os, sys, time, logging
import numpy as np, pandas as pd, scipy.sparse as sp, torch, anndata as ad, scvi

from txcellnet.bench import (load_kidney, group_genes, group_cases, load_case_cells as load_case, subset_genes, score_cross,
                             within_split, write_within, xsec_train, xsec_test, write_xsec, PROTS, log)

SEED = int(os.environ.get('SEED', 1)); scvi.settings.seed = SEED
N_REF, N_QUERY = int(os.environ.get('N_REF', 100_000)), int(os.environ.get('N_QUERY', 100_000))
torch.set_float32_matmul_precision('high'); logging.getLogger('lightning').setLevel(logging.WARNING)


def sub(idx, n, seed=0):
    return idx if len(idx) <= n else np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))


def fit_predict(refs, qname, Xq_train, Xq_pred, genes, names):
    """refs: [(batch, counts csr, protein counts)]; query batch qname: training counts Xq_train, cells to predict Xq_pred.
    Returns denoised protein (Xq_pred rows x names), expected counts."""
    Xs = [r[1] for r in refs] + [Xq_train]; b = np.concatenate([[r[0]] * r[1].shape[0] for r in refs] + [[qname] * Xq_train.shape[0]])
    lib = np.asarray(sp.vstack(Xs).sum(1)).ravel() > 0
    cats = [r[0] for r in refs] + [qname]
    A = ad.AnnData(sp.vstack(Xs).tocsr()[lib].toarray().astype(np.float32), obs=pd.DataFrame({'batch': pd.Categorical(b[lib], categories=cats)},
                                                                                index=[f'c{i}' for i in range(int(lib.sum()))]))
    A.var_names = genes
    keep = np.ones(A.n_obs, bool)
    if A.n_vars > 4000:  # the tutorial's gene selection (panels with <= 4,000 genes keep all of them)
        import scanpy as sc
        sc.pp.highly_variable_genes(A, batch_key='batch', flavor='seurat_v3', n_top_genes=4000, subset=True)
        keep = np.asarray(A.X.sum(1)).ravel() > 0; A = A[keep].copy()  # library size > 0 on the selected genes
    gi = [{g: i for i, g in enumerate(genes)}[g] for g in A.var_names]
    P = np.vstack([r[2] for r in refs] + [np.zeros((Xq_train.shape[0], len(names)))]).astype(np.float32)[lib][keep]
    A.obsm['protein_expression'] = pd.DataFrame(P, index=A.obs_names, columns=names)
    scvi.model.TOTALVI.setup_anndata(A, batch_key='batch', protein_expression_obsm_key='protein_expression')
    model = scvi.model.TOTALVI(A, latent_distribution='normal', n_layers_decoder=2)
    ep = int(os.environ.get('MAX_EPOCHS', 0)) or min(round(20000 / A.n_obs * 400), 400); t0 = time.time()
    model.train(max_epochs=ep, enable_progress_bar=False)
    hist = model.history; last = {k: float(v.iloc[-1, 0]) for k, v in hist.items() if k in ('elbo_train', 'elbo_validation')}
    log('totalVI trained', A.n_obs, 'cells', A.n_vars, 'genes', ep, 'max epochs', len(hist['elbo_train']), 'run', last, f'{time.time() - t0:.0f}s')
    B = ad.AnnData(Xq_pred.tocsr()[:, gi].astype(np.float32), obs=pd.DataFrame({'batch': pd.Categorical([qname] * Xq_pred.shape[0], categories=cats)},
                                                                        index=[f'q{i}' for i in range(Xq_pred.shape[0])]))
    B.var_names = list(A.var_names)
    B.obsm['protein_expression'] = pd.DataFrame(np.zeros((B.n_obs, len(names)), np.float32), index=B.obs_names, columns=names)
    _, prot = model.get_normalized_expression(B, n_samples=25, return_mean=True, transform_batch=[r[0] for r in refs],
                                              include_protein_background=True, sample_protein_mixing=False,
                                              gene_list=[A.var_names[0]], batch_size=1024)
    return np.asarray(prot, np.float64)


def run_cross(OUT, groups):
    os.makedirs(f'{OUT}/preds', exist_ok=True); kid = load_kidney(); rp = f'{OUT}/bench_rows.csv'
    rows = pd.read_csv(rp).to_dict('records') if os.path.exists(rp) else []
    for gname in groups:
        genes = group_genes(gname, kid)
        refs = []; L = []
        for k in kid:
            if not k['tr'].any(): continue  # KID_TRAIN: the other section has no training cells
            t = sub(np.flatnonzero(k['tr']), N_REF); refs.append((k['name'], subset_genes(k['G'], k['genes'], genes)[t], k['P'][t]))
            L.append(np.log1p(k['P'][t].astype(np.float64)))
        L = np.vstack(L); mu, sd = L.mean(0), L.std(0) + 1e-6
        for c in group_cases(gname):
            case = load_case(c); Xq = subset_genes(case['X'], case['genes'], genes); ok = np.flatnonzero(case['ok'])
            prot = fit_predict(refs, 'query', Xq[sub(ok, N_QUERY)], Xq[ok], genes, PROTS)
            pz = np.full((len(case['cells']), len(PROTS)), np.nan, np.float32); pz[ok] = (np.log1p(prot) - mu) / sd
            np.save(f'{OUT}/preds/{c}__totalvi_pm.npy', pz.astype(np.float16))
            rows = [r for r in rows if not (r['case'] == case['meta']['dataset'] and r['method'] == 'totalvi_pm')]
            rows += score_cross(case, pz, 'totalvi_pm', gname, len(genes))
            df = pd.DataFrame(rows); df.to_csv(rp, index=False)
            log(c, gname, 'totalvi_pm median Spearman', round(float(df[df.case == case['meta']['dataset']].spearman.median()), 3))


def run_within(cases):
    for c in cases:
        case = load_case(c); tr, te, keys, T, R = within_split(case)
        genes = list(case['genes']); X = case['X']  # all panel genes; fit_predict applies the tutorial's HVG selection
        s = 500.0 / np.maximum(np.percentile(R[tr], 99, axis=0), 1e-6); Pc = np.round(R * s)
        trs = sub(tr, N_REF * 2)  # <= 200,000 training-stripe cells
        log(c, 'train', len(tr), '(used', len(trs), ') test', len(te), 'genes', len(genes), 'targets', keys)
        prot = fit_predict([('train', X[trs], Pc[trs])], 'test', X[sub(te, N_QUERY)], X[te], genes, [f'p{j}' for j in range(len(keys))])
        pred = np.full((len(case['cells']), len(keys)), np.nan, np.float32); pred[te] = np.log1p(prot)
        write_within(c, 'totalvi', keys, pred, T, te)


def run_xsec(a, bs):
    A, trA, kA, TA, RA = xsec_train(a); genes = list(A['genes'])  # all panel genes; fit_predict applies the tutorial's HVGs
    s = 500.0 / np.maximum(np.percentile(RA[trA], 99, axis=0), 1e-6); Pc = np.round(RA * s)
    trs = sub(trA, N_REF * 2)  # <= 200,000 training-stripe cells, as in within mode
    for b in bs:
        B, keys, ia, teB, okB, TB = xsec_test(b, kA); XB = subset_genes(B['X'], B['genes'], genes)
        log(a, '->', b, 'reference', len(trs), 'query', len(okB), 'genes', len(genes), 'targets', keys)
        prot = fit_predict([('reference', A['X'][trs], Pc[trs])], 'query', XB[sub(okB, N_QUERY)], XB[okB], genes,
                           [f'p{j}' for j in range(len(kA))])
        pred = np.full((len(B['cells']), len(kA)), np.nan, np.float32); pred[okB] = np.log1p(prot)
        write_xsec(a, b, 'totalvi', keys, pred[:, ia], TB, teB, okB)


if __name__ == '__main__':
    if sys.argv[1] == 'cross': run_cross(sys.argv[2], sys.argv[3].split(','))
    elif sys.argv[1] == 'within': run_within(sys.argv[2:])
    elif sys.argv[1] == 'xsec': run_xsec(sys.argv[2], sys.argv[3].split(','))
    else: raise SystemExit(__doc__)
