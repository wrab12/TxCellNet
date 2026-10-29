"""NicheTrans exactly as in its official Xenium tutorial (github.com/YSTLab/NicheTrans, commit 85fab47:
Tutorial_7.1__Train_NicheTrans_on_10x_Xenium_data.ipynb, args/args_breast_cancer.py, datasets/data_manager_breast_cancer.py,
utils/utils_training_breast_cancer.py, model/nicheTrans.py):
  RNA       normalize_total(1e3) + log1p over all genes the model reads
  targets   log(x + 1) of the per-cell protein intensity (no further scaling)
  niche     12 nearest spatial neighbours of each cell (excluding itself), sorted by distance; the graph is built within the
            training cells and, separately, within the cells to be predicted (as _process_data on the two halves)
  model     NicheTrans(noise_rate=0.2, dropout_rate=0.2); MSE loss
  training  Adam (lr 3e-4, weight decay 5e-4), StepLR(step 20, gamma 0.1), 40 epochs, batch 32 (shuffled, drop_last);
            with probability 0.3 a batch's neighbour features are masked by Bernoulli(0.5) per neighbour; last epoch is used
Only the execution differs: data are GPU tensors instead of a DataLoader of Python lists (identical batches), and each
optimizer step is replayed as a CUDA graph of the identical computation (nt_graph_equiv.py: graph vs eager differences are at
floating-point rounding level, smaller than those caused by perturbing the inputs by 1e-7).
Predictions are in the method's log(x+1) units; cross-dataset predictions are mapped to kidney z-units with the training
targets' mean and SD (an affine map per protein, as for DGAT) so that stains are combined with the kidney-fitted weights.
modes:
  cross OUT_DIR GROUP[,GROUP...]   kidney training stripes, the group's genes measured in both sections (panel-matched as the
                                   other cell-level methods; GENESET=union: all genes of the group, zero where a section lacks
                                   one, suffix _u); SEED (default 1, suffix _s<seed>) -> method nichetrans_off[_u][_s<seed>]
                                   VHD=1: the group's Visium HD sections instead of its Xenium cases (Supplementary S4)
  within CASE [CASE ...]           training stripes of the case, all panel genes -> results/bench/within/CASE__nichetrans_off.csv
  xsec A B[,B...]                  cross-section tier: trained as in within mode on case A, then predicts every evaluated cell
                                   of each case B of the same panel group (graph within those cells); scored on B's held-out
                                   stripes and on all its cells -> results/bench/xsec/A__B__nichetrans_off.csv
                                   (kidney sections: cross mode with KID_TRAIN=rcc|ccrcc, see txcellnet.bench.load_kidney)
  sim PERT                         semi-synthetic benchmark (sim_sota.py prep cache) -> results/sim/sota/pred_nichetrans_PERT.npz
NT_EPOCHS / NT_TAG: smoke tests only (fewer epochs, tagged outputs).
The official model/nicheTrans.py imports torchvision and einops (the paper: torchvision 0.20.1, einops 0.8.2).
"""
import json, os, random, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, torch, torch.nn as nn
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.environ.get('NICHETRANS_DIR', 'external/NicheTrans'))
from model.nicheTrans import NicheTrans

dev = 'cuda'; SEED = int(os.environ.get('SEED', 1)); K_NB = 12
EPOCHS = int(os.environ.get('NT_EPOCHS', 40)); TAG = os.environ.get('NT_TAG', '')


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def lognorm(X):
    """sc.pp.normalize_total(target_sum=1e3) + sc.pp.log1p (cells without counts stay zero) -> csr float32."""
    X = X.astype(np.float32).tocsr(); tot = np.asarray(X.sum(1)).ravel()
    A = (sp.diags(np.where(tot > 0, 1e3 / np.maximum(tot, 1e-12), 0).astype(np.float32)) @ X).tocsr(); A.data = np.log1p(A.data)
    return A


def knn(xy):
    """Cal_Spatial_Net_row_col(k_cutoff=12, model='KNN'): 12 nearest neighbours, excluding the cell itself, nearest first."""
    return NearestNeighbors(n_neighbors=K_NB + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]


def train_nt(F, Y, nb, seed=SEED, epochs=EPOCHS):
    """Tutorial 7.1 loop with utils_training_breast_cancer.train. F: features of the training cells, Y: log(x+1) targets,
    nb: neighbour indices into F (all float32 / int64 on the GPU)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)  # utils.set_seed
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model = NicheTrans(source_length=F.shape[1], target_length=Y.shape[1], noise_rate=0.2, dropout_rate=0.2).to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4, capturable=True)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1); crit = nn.MSELoss()
    idx = torch.zeros(32, dtype=torch.long, device=dev); mask = torch.ones(32, K_NB, 1, device=dev)

    def step():  # neighbour features times the batch's mask (all ones when the batch is not masked; x * 1 == x)
        loss = crit(model(F[idx], F[nb[idx]] * mask), Y[idx]); loss.backward(); opt.step(); return loss.detach()

    nbat = len(F) // 32; graph = out = glr = None; t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(F), device=dev); tot = torch.zeros((), device=dev)  # shuffle=True, drop_last=True
        for i in range(nbat):
            idx.copy_(perm[i * 32:(i + 1) * 32])
            if random.random() > 0.7: mask.copy_(torch.bernoulli(torch.full((32, K_NB, 1), 0.5, device=dev)))
            else: mask.fill_(1.0)
            if ep == 0 and i < 3:  # eager warm-up steps (side stream) before the first capture
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    opt.zero_grad(set_to_none=True); tot += step()
                torch.cuda.current_stream().wait_stream(s); continue
            if graph is None or glr != opt.param_groups[0]['lr']:  # (re)capture when StepLR changes the learning rate
                graph = out = None; opt.zero_grad(set_to_none=True); glr = opt.param_groups[0]['lr']; graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = step()
            graph.replay(); tot += out
        sched.step()
        if ep == 0 or (ep + 1) % 5 == 0: log(f'epoch {ep + 1}/{epochs} loss {tot.item() / nbat:.4f} lr {glr:.0e} {time.time() - t0:.0f}s')
    del graph; torch.cuda.synchronize()
    return model.eval()


@torch.no_grad()
def predict_nt(model, F, nb, bs=4096):
    return np.vstack([model(F[i:i + bs], F[nb[i:i + bs]]).cpu().numpy() for i in range(0, len(F), bs)])


def gpu(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev)


def main():
    MODE = sys.argv[1]
    if MODE == 'cross':
        from txcellnet.bench import (load_kidney, group_genes, group_cases, load_case_cells as load_case, subset_genes, score_cross,
                                     PANELS, PROTS)
        OUT, GROUPS = sys.argv[2], sys.argv[3].split(','); os.makedirs(f'{OUT}/preds', exist_ok=True)
        union = os.environ.get('GENESET', 'both') == 'union'
        mname = 'nichetrans_off' + TAG + ('_u' if union else '') + (f'_s{SEED}' if SEED != 1 else ''); rp = f'{OUT}/bench_rows.csv'
        rows = pd.read_csv(rp).to_dict('records') if os.path.exists(rp) else []
        kid = load_kidney()
        for gname in GROUPS:
            genes = list(PANELS[gname]['genes']) if union else group_genes(gname, kid); Fs, Ys, NBs, off = [], [], [], 0
            for k in kid:
                tr = np.flatnonzero(k['tr'])
                if not len(tr): continue  # KID_TRAIN: the other section has no training cells
                Fs.append(lognorm(subset_genes(k['G'], k['genes'], genes)[tr]).toarray())
                Ys.append(np.log1p(k['P'][tr].astype(np.float32))); NBs.append(knn(k['xy'][tr]) + off); off += len(tr)
            Ftr, Ytr = np.vstack(Fs), np.vstack(Ys); mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6
            log(gname, mname, 'genes', len(genes), 'training cells', len(Ftr))
            model = train_nt(gpu(Ftr), gpu(Ytr), gpu(np.vstack(NBs))); del Fs, Ftr
            vhd = [c for c in PANELS[gname]['cases'] if c.startswith('vhd') and os.path.exists(f'data/cases/{c}/meta.json')]
            for c in (vhd if os.environ.get('VHD') == '1' else group_cases(gname)):
                case = load_case(c); cells = case['cells']
                # cells to predict = the test set: kidney held-out stripes; other datasets: every cell with >= 1 count
                node = np.flatnonzero(case['ok']) if c.startswith('kidney') else \
                    np.flatnonzero((cells['area_px'].values >= 10) & (np.asarray(case['X'].sum(1)).ravel() >= 1))
                Fq = gpu(lognorm(subset_genes(case['X'], case['genes'], genes)[node]).toarray())
                pz = np.full((len(cells), len(PROTS)), np.nan, np.float32)
                pz[node] = (predict_nt(model, Fq, gpu(knn(case['xy'][node]))) - mu) / sd; del Fq
                np.save(f'{OUT}/preds/{c}__{mname}.npy', pz.astype(np.float16))
                rows = [r for r in rows if not (r['case'] == case['meta']['dataset'] and r['method'] == mname)]
                rows += score_cross(case, pz, mname, gname, len(genes)); df = pd.DataFrame(rows); df.to_csv(rp, index=False)
                log(c, mname, 'median Spearman', round(float(df[(df.case == case['meta']['dataset']) & (df.method == mname)].spearman.median()), 3))
            del model; torch.cuda.empty_cache()
    elif MODE == 'within':
        from txcellnet.bench import load_case_cells as load_case, within_split, write_within
        for c in sys.argv[2:]:
            case = load_case(c); tr, te, keys, T, R = within_split(case)
            A = lognorm(case['X'])  # all panel genes, as the tutorial (no gene selection)
            log(c, 'train', len(tr), 'test', len(te), 'genes', A.shape[1], 'targets', keys)
            model = train_nt(gpu(A[tr].toarray()), gpu(np.log1p(R[tr]).astype(np.float32)), gpu(knn(case['xy'][tr])))
            torch.cuda.empty_cache(); pred = np.full((len(case['cells']), len(keys)), np.nan, np.float32)
            Fq = gpu(A[te].toarray()); pred[te] = predict_nt(model, Fq, gpu(knn(case['xy'][te]))); del Fq
            write_within(c, 'nichetrans_off' + TAG, keys, pred, T, te)
            del model; torch.cuda.empty_cache()
    elif MODE == 'xsec':
        from txcellnet.bench import xsec_train, xsec_test, write_xsec, subset_genes
        a = sys.argv[2]; A, trA, kA, TA, RA = xsec_train(a)
        X = lognorm(A['X'])  # all panel genes of A, as in within mode
        log(a, 'xsec train', len(trA), 'genes', X.shape[1], 'targets', kA)
        model = train_nt(gpu(X[trA].toarray()), gpu(np.log1p(RA[trA]).astype(np.float32)), gpu(knn(A['xy'][trA]))); del X
        torch.cuda.empty_cache()
        for b in sys.argv[3].split(','):
            B, keys, ia, teB, okB, TB = xsec_test(b, kA)
            Fq = gpu(lognorm(subset_genes(B['X'], B['genes'], A['genes'])[okB]).toarray())
            pred = np.full((len(B['cells']), len(keys)), np.nan, np.float32)
            pred[okB] = predict_nt(model, Fq, gpu(knn(B['xy'][okB])))[:, ia]; del Fq
            write_xsec(a, b, 'nichetrans_off' + TAG, keys, pred, TB, teB, okB)
        del model; torch.cuda.empty_cache()
    elif MODE == 'sim':
        PERT = sys.argv[2]; CACHE = 'data/sim/sota_cache'; SECT = ['rcc', 'ccrcc']
        S = {s: dict(X=sp.load_npz(f'{CACHE}/{PERT}_{s}_X.npz').tocsr(), **dict(np.load(f'{CACHE}/{PERT}_{s}.npz'))) for s in SECT}
        Fs, Ys, NBs, off = [], [], [], 0
        for s in SECT:
            tr = np.flatnonzero(S[s]['train']); Fs.append(lognorm(S[s]['X'][tr]).toarray())
            Ys.append(np.log1p(np.clip(S[s]['tgt'][tr], 0, None)).astype(np.float32)); NBs.append(knn(S[s]['xy'][tr]) + off); off += len(tr)
        log('sim', PERT, 'training cells', off, 'genes', Fs[0].shape[1])
        model = train_nt(gpu(np.vstack(Fs)), gpu(np.vstack(Ys)), gpu(np.vstack(NBs))); del Fs
        pred = {}
        for s in SECT:  # training-stripe cells with the training graph, held-out cells with a graph of their own
            p = np.full((len(S[s]['node']), S[s]['tgt'].shape[1]), np.nan, np.float32)
            for part in (S[s]['train'], S[s]['node'] & ~S[s]['train']):
                q = np.flatnonzero(part); p[q] = predict_nt(model, gpu(lognorm(S[s]['X'][q]).toarray()), gpu(knn(S[s]['xy'][q])))
            pred[s] = p
        os.makedirs('results/sim/sota', exist_ok=True); np.savez(f'results/sim/sota/pred_nichetrans{TAG}_{PERT}.npz', **pred); log('sim', PERT, 'saved')
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
