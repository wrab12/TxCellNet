"""Spot-level ("grid") evaluation: DGAT was introduced for Visium spots, so it is also evaluated in that setting, with every
method scored at the same spot resolution.

Pseudo-spots: the cells of a section are grouped by a hexagonal lattice whose neighbouring centres are PITCH um apart (default
55 um, so a spot covers about the area of a 55-um Visium spot; the spots tile the tissue). A spot that straddles a stripe
boundary or the edge of the evaluated area is split there, so each spot holds only training, only test or only unscored
cells. A spot's RNA is the summed counts of its cells; its protein value is the mean over its cells of the per-cell target
(log1p intensity, as scored at the cell level).

Methods (Spearman across the test spots, per marker):
  dgat_spot   DGAT trained and applied on the pseudo-spots, with the functions of dgat_bench.py (official model, losses,
              optimisers, early stopping, preprocessing - normalize_total, log1p and scaling per sample; protein CLR, here of
              the spot intensity expm1(mean log1p) - and graphs: spatial kNN k=6 on the spot centroids (mean position of the
              spot's cells; mostly the six neighbours of the hexagonal lattice, as for Visium), united with an
              expression kNN k=10 and self loops; protein kNN). Training graphs hold training spots only (tiles of about
              TILE_TRAIN spots); prediction graphs hold all spots of the section (tiles of about TILE_PRED spots).
  dgat_cell, nichetrans, txcellnet   the per-cell predictions of the main benchmark, averaged over the cells of each spot.
Cell-level scores on the same test cells are written too (level='cell') to check them against the main benchmark.

modes:
  within CASE [CASE ...]   same-section setting of the 32 external sections -> OUT/within/CASE.csv
  cross GROUP[,GROUP...]   DGAT trained on the kidney training stripes with the group's genes, as dgat_bench.py cross mode
                           -> OUT/cross/CASE.csv (group 'full' = held-out stripes of the two kidney sections, which are
                           the same-section setting of the training tissue)
Summary tables: scripts/evaluation/grid_summary.py.
env: PITCH (um, default 55), OUT (default results/bench/grid[_<PITCH>um]), SEED (default 1), TILE_TRAIN, TILE_PRED
"""
import glob, json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp

PITCH = float(os.environ.get('PITCH', 55))
OUT = os.environ.get('OUT', 'results/bench/grid' + ('' if PITCH == 55 else f'_{int(PITCH)}um'))
CACHE = 'results/bench/predcache'


def hex_ids(xy, d=PITCH):
    """(sub-lattice, i, j) of the nearest centre of a hexagonal lattice whose neighbouring centres are d apart
    (two rectangular sub-lattices offset by half a period, as in matplotlib's hexbin)."""
    h = d * np.sqrt(3.0)
    a = np.stack([np.round(xy[:, 0] / d), np.round(xy[:, 1] / h)], 1)
    b = np.stack([np.round(xy[:, 0] / d - 0.5), np.round(xy[:, 1] / h - 0.5)], 1)
    da = ((xy - a * [d, h]) ** 2).sum(1); db = ((xy - (b + 0.5) * [d, h]) ** 2).sum(1)
    useb = db < da
    return np.column_stack([useb, np.where(useb[:, None], b, a)]).astype(np.int64)


def spots(xy, part):
    """spot index of every cell (rows of xy): hexagonal bins split by `part` (integer label per cell);
    returns (index per cell, number of spots, part of each spot)."""
    _, inv = np.unique(np.column_stack([hex_ids(xy), part.astype(np.int64)]), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel(); ns = int(inv.max()) + 1
    sp_part = np.zeros(ns, np.int64); sp_part[inv] = part
    return inv, ns, sp_part


def member(inv, ns):
    return sp.csr_matrix((np.ones(len(inv), np.float32), (inv, np.arange(len(inv)))), shape=(ns, len(inv)))


def agg_sum(X, inv, ns):
    return (member(inv, ns) @ X.tocsr()).tocsr()


def agg_mean(A, inv, ns):
    """mean over the cells of each spot, ignoring non-finite values (NaN where a spot has none)."""
    A = np.asarray(A, np.float64); A = A[:, None] if A.ndim == 1 else A; fin = np.isfinite(A); M = member(inv, ns)
    s = M @ np.where(fin, A, 0.0); c = M @ fin.astype(np.float64)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def run_within(cases):
    import torch
    from txcellnet.bench import load_case_cells as load_case, within_split, subset_genes
    from txcellnet.metrics import metrics
    from txcellnet.utils import log
    from dgat_bench import rna_scale, clr0, dgat_genes, tiles, tile_graph, train_dgat, predict, TILE_TRAIN
    os.makedirs(f'{OUT}/within', exist_ok=True)
    for c in cases:
        case = load_case(c); tr, te, keys, T, _ = within_split(case); cells = case['cells']; cd = f'data/cases/{c}'
        node = np.flatnonzero((cells['area_px'].values >= 10) & (np.asarray(case['X'].sum(1)).ravel() >= 1))
        part = np.full(len(cells), 2); part[tr] = 0; part[te] = 1
        inv, ns, spart = spots(case['xy'][node], part[node]); trs, tes = np.flatnonzero(spart == 0), np.flatnonzero(spart == 1)
        ncell = np.bincount(inv, minlength=ns); pos = agg_mean(case['xy'][node], inv, ns)
        Xs = agg_sum(case['X'][node], inv, ns); Ts = agg_mean(T[node], inv, ns)
        own = {g for k in keys for g in case['meta']['markers'][k].get('own', [])}
        genes = dgat_genes(Xs, case['genes'], trs, own)
        Xr = rna_scale(subset_genes(Xs, case['genes'], genes))
        Pc = clr0(np.expm1(np.clip(Ts[trs], 0, None))); mu, sd = Pc.mean(0), Pc.std(0) + 1e-6
        graphs = [tile_graph(Xr[trs[t]], pos[trs[t]], Pc[t]) for t in tiles(pos[trs], TILE_TRAIN)]
        log(c, 'spots', ns, 'train', len(trs), 'test', len(tes), 'cells/test spot (median)', float(np.median(ncell[tes])),
            'genes', len(genes), 'targets', keys, 'graphs', len(graphs))
        enc_r, dec_p = train_dgat(graphs, len(genes), len(keys)); del graphs
        P = {'dgat_spot': (predict(enc_r, dec_p, Xr, pos, len(keys)) - mu) / sd}
        del enc_r, dec_p; torch.cuda.empty_cache()
        rows = []; yj = json.load(open(f'{cd}/Y.json'))
        for m, f, ref in (('txcellnet', f'{cd}/runs/within_clean/cellpred.npy', 'within_clean'),
                          ('dgat_cell', f'{cd}/runs/within_baselines/dgat.npy', 'dgat'),
                          ('nichetrans', f'{cd}/runs/within_baselines/nichetrans_off.npy', 'nichetrans_off')):
            pc = np.load(f).astype(np.float64)
            if m == 'txcellnet': pc = pc[:, [yj['channels'].index(yj['markers'][k]) for k in keys]]  # train_case.py: columns = Y.json channels
            assert pc.shape == (len(cells), len(keys)), (m, pc.shape)
            P[m] = agg_mean(pc[node], inv, ns)
            rc = [dict(case=c, marker=k, method=m, level='cell', **metrics(pc[te, j], T[te, j]), n=len(te)) for j, k in enumerate(keys)]
            got = pd.DataFrame(rc).set_index('marker').spearman
            want = pd.read_csv(f'results/bench/within/{c}__{ref}.csv').set_index('marker').spearman.reindex(got.index)
            assert np.allclose(got, want, atol=1e-6, equal_nan=True), (m, 'cell-level scores differ from the main benchmark', got.round(4).to_dict(), want.round(4).to_dict())
            rows += rc
        for m, ps in P.items():
            rows += [dict(case=c, marker=k, method=m, level='spot', **metrics(ps[tes, j], Ts[tes, j]), n=len(tes)) for j, k in enumerate(keys)]
        df = pd.DataFrame(rows).assign(pitch=PITCH, spots_test=len(tes), cells_per_spot=float(np.median(ncell[tes])), dgat_genes=len(genes))
        df.to_csv(f'{OUT}/within/{c}.csv', index=False)
        log(c, 'median Spearman', df.groupby(['level', 'method']).spearman.median().round(3).to_dict())


def tx_cross(c):
    """TxCellNet per-cell predictions of the main benchmark: kidney held-out stripes = one clean network (fin_clean_s1),
    other datasets = the six-network ensemble (mean of the cached single-network predictions, as run_bench_gen.py ENS)."""
    if c.startswith('kidney'): return np.load(f'{CACHE}/{c}__fin_clean_s1.npy').astype(np.float64)
    return np.mean([np.load(f'{CACHE}/{c}__fin_{r}_s{s}.npy').astype(np.float64) for r in ('clean', 'rand') for s in (1, 2, 3)], 0)


def nt_cross(c, g):
    """NicheTrans: kidney = nichetrans_off_u; other datasets = the three-seed ensemble (ens_cellbase.py: mean of the seeds)."""
    if c.startswith('kidney'): return np.load(f'results/bench/nt_off_full_u/preds/{c}__nichetrans_off_u.npy').astype(np.float64)
    fs = sorted(f for d in glob.glob(f'results/bench/nt_off_{g}_s[123]')  # file names as in ens_cellbase.py: <case>__<m>[_s<k>].npy
                for f in glob.glob(f'{d}/preds/{c}__nichetrans_off.npy') + glob.glob(f'{d}/preds/{c}__nichetrans_off_s[0-9].npy'))
    assert len(fs) == 3, (c, g, fs)
    return np.mean([np.load(f).astype(np.float64) for f in fs], 0)


def run_cross(groups):
    import torch
    from txcellnet.bench import load_kidney, group_genes, group_cases, load_case_cells as load_case, subset_genes, combine, PROTS, OWN
    from txcellnet.metrics import metrics
    from txcellnet.utils import log
    from dgat_bench import rna_scale, clr0, dgat_genes, tiles, tile_graph, train_dgat, predict, TILE_TRAIN
    os.makedirs(f'{OUT}/cross', exist_ok=True); kid = load_kidney(); own = {g for p in PROTS for g in OWN.get(p, [])}
    for G in groups:
        genes0 = group_genes(G, kid); blocks = []
        for k in kid:
            tri = np.flatnonzero(k['tr'])
            if not len(tri): continue
            inv, ns, _ = spots(k['xy'][tri], np.zeros(len(tri), np.int64))
            blocks.append((agg_sum(subset_genes(k['G'], k['genes'], genes0)[tri], inv, ns),
                           np.expm1(agg_mean(np.log1p(np.clip(k['P'][tri].astype(np.float64), 0, None)), inv, ns)),
                           agg_mean(k['xy'][tri], inv, ns)))
        Gtr = sp.vstack([b[0] for b in blocks]).tocsr(); genes = dgat_genes(Gtr, genes0, np.arange(Gtr.shape[0]), own)
        graphs, Ptr = [], []
        for Xs, Ps, pos in blocks:
            Xr = rna_scale(subset_genes(Xs, genes0, genes)); Pc = clr0(Ps); Ptr.append(Pc)
            graphs += [tile_graph(Xr[t], pos[t], Pc[t]) for t in tiles(pos, TILE_TRAIN)]
        Ptr = np.vstack(Ptr); mu, sd = Ptr.mean(0), Ptr.std(0) + 1e-6
        log(G, 'genes', len(genes), 'of', len(genes0), 'training spots', len(Ptr), 'graphs', len(graphs))
        enc_r, dec_p = train_dgat(graphs, len(genes), len(PROTS)); del graphs
        for c in group_cases(G):
            case = load_case(c); cells = case['cells']; meta = case['meta']; ok = np.flatnonzero(case['ok'])
            node = np.flatnonzero((cells['area_px'].values >= 10) & (np.asarray(case['X'].sum(1)).ravel() >= 1))
            part = np.where(case['ok'], 1, 2)
            inv, ns, spart = spots(case['xy'][node], part[node]); tes = np.flatnonzero(spart == 1); ncell = np.bincount(inv, minlength=ns)
            pos = agg_mean(case['xy'][node], inv, ns)
            pz = (predict(enc_r, dec_p, rna_scale(agg_sum(subset_genes(case['X'], case['genes'], genes)[node], inv, ns)), pos, len(PROTS)) - mu) / sd
            cellp = {'txcellnet': tx_cross(c), 'nichetrans': nt_cross(c, G),
                     'dgat_cell': np.load(f'results/bench/dgat_{G}/preds/{c}__dgat_pm.npy').astype(np.float64)}
            rows = []
            for key, mk in meta['markers'].items():
                tt = cells[mk['truth']].values.astype(np.float64)
                if mk.get('log', True): tt = np.log1p(np.clip(tt - mk.get('offset', 0), 0, None))
                ts = agg_mean(tt[node], inv, ns)[:, 0]
                for m, pc in cellp.items():
                    v = combine(pc, mk['components'], key)
                    rows.append(dict(case=c, marker=key, method=m, level='cell', **metrics(v[ok], tt[ok]), n=len(ok)))
                    rows.append(dict(case=c, marker=key, method=m, level='spot', **metrics(agg_mean(v[node], inv, ns)[tes, 0], ts[tes]), n=len(tes)))
                rows.append(dict(case=c, marker=key, method='dgat_spot', level='spot', **metrics(combine(pz, mk['components'], key)[tes], ts[tes]), n=len(tes)))
            df = pd.DataFrame(rows).assign(group=G, pitch=PITCH, spots_test=len(tes), cells_per_spot=float(np.median(ncell[tes])), dgat_genes=len(genes))
            df.to_csv(f'{OUT}/cross/{c}.csv', index=False)
            log(c, 'spots', ns, 'test', len(tes), 'median Spearman', df.groupby(['level', 'method']).spearman.median().round(3).to_dict())
        del enc_r, dec_p; torch.cuda.empty_cache()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'within': run_within(sys.argv[2:])
    elif mode == 'cross': run_cross(sys.argv[2].split(','))
    else: raise SystemExit(__doc__)
