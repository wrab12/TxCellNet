"""Add all-gene molecules to prepared benchmark cases (tx_all.npz + genes_all.json), for gene-panel-agnostic models.

The original tx.npz keeps only genes of the kidney model vocabulary. Here every real gene of the case panel that has a
GenePT vector is kept, on exactly the same grid as the case (same origin / crop / pixel size).
Gene names are handled as integer codes throughout (5K / whole-transcriptome cases have 10^8 molecules).
usage: add_allgenes.py CASE [CASE ...]      (case names under data/cases/)
"""
import glob, json, sys, time
import numpy as np, pandas as pd, h5py, scipy.sparse as sp

from txcellnet.genes import genept, ALIAS

UM = 0.85
CTRL = ('NegControl', 'BLANK', 'Unassigned', 'Deprecated', 'Intergenic', 'antisense')
CRLM = {'crlm_tumor1': 'GSM9815454_tumor1_M242T', 'crlm_tumor2': 'GSM9815455_tumor2_M131T',
        'crlm_adjnormal': 'GSM9815457_AdjNormal_M131N', 'crlm_tumor3': 'GSM9815456_tumor3_M397T'}


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def save(cd, gy, gx, codes, cats, H, W):
    """codes: int index into cats per molecule."""
    gp = genept()
    genes = sorted({c for c in cats if not c.startswith(CTRL) and ALIAS.get(c, c).upper() in gp})
    pos = {g: i for i, g in enumerate(genes)}
    c2i = np.array([pos.get(c, -1) for c in cats], np.int64)
    gi = c2i[codes]; k = gi >= 0
    gy = np.clip(gy[k].astype(np.int64), 0, H - 1).astype(np.int32); gx = np.clip(gx[k].astype(np.int64), 0, W - 1).astype(np.int32)
    np.savez(f'{cd}/tx_all.npz', gy=gy, gx=gx, gidx=gi[k].astype(np.int32)); json.dump(genes, open(f'{cd}/genes_all.json', 'w'))
    log(cd, 'all-gene molecules', int(k.sum()), 'genes', len(genes))


def xenium_tx(path, qv=20):
    tx = pd.read_parquet(path, columns=['x_location', 'y_location', 'feature_name', 'qv'])
    ok = tx['qv'].values >= qv
    col = tx['feature_name']
    if isinstance(col.dtype, pd.CategoricalDtype):
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in col.cat.categories]; codes = col.cat.codes.values.astype(np.int64)
    else:
        v = col.values
        v = np.array([s.decode() for s in v]) if len(v) and isinstance(v[0], bytes) else v.astype(str)
        cats, codes = np.unique(v, return_inverse=True); cats = list(cats)
    return tx['x_location'].values[ok].astype(np.float32), tx['y_location'].values[ok].astype(np.float32), codes[ok], cats


for case in sys.argv[1:]:
    cd = f'data/cases/{case}'; meta = json.load(open(f'{cd}/meta.json')); H, W = meta['H'], meta['W']
    if case.startswith('xen_') or case.startswith('kidney_'):
        outs = {'kidney_rcc': 'data/xenium_protein_rcc/outs', 'kidney_ccrcc': 'data/xenium_protein_ccrcc/outs', 'xen_breast_5k': 'data/pre/pre_breast_5k', 'xen_cervix_5k': 'data/pre/pre_cervix_5k', 'xen_ovary_5k': 'data/pre/pre_ovary_5k', 'xen_prostate_5k': 'data/pre/pre_prostate_5k', 'xen_skin_5k': 'data/pre/pre_skin_5k', 'xen_gbm_v1': 'data/pre/pre_gbm_v1', 'xen_ovary_v1ff': 'data/pre/pre_ovary_v1_ff'}.get(case, f'data/ext/{case}')
        x, y, codes, cats = xenium_tx(f'{outs}/transcripts.parquet')
        save(cd, y / UM, x / UM, codes, cats, H, W)
    elif case.startswith('crlm_'):
        smp = CRLM['crlm_tumor3' if case.startswith('crlm_tumor3') else case]
        y0, y1, x0, x1 = meta['crop']
        x, y, codes, cats = xenium_tx(f'data/crlm_xenium_imc/xenium/ex/{smp}_transcripts.parquet')
        k = (x / UM >= x0) & (x / UM < x1) & (y / UM >= y0) & (y / UM < y1)
        save(cd, y[k] / UM - y0, x[k] / UM - x0, codes[k], cats, H, W)
    elif case.startswith('duchini_'):
        core = case[len('duchini_'):]
        x, y, codes, cats = xenium_tx(f'data/duchini_xenium_comet/ex/UnumLocalia/{core}/xenium/transcripts.parquet')
        x0, y0 = x.min() - 20, y.min() - 20
        save(cd, (y - y0) / UM, (x - x0) / UM, codes, cats, H, W)
    elif case.startswith('vhd_'):
        b2 = glob.glob(f'data/ext/{case}/**/square_002um', recursive=True)[0]
        mpp = float(json.load(open(f'{b2}/spatial/scalefactors_json.json'))['microns_per_pixel'])
        with h5py.File(f'{b2}/filtered_feature_bc_matrix.h5', 'r') as f:
            m = f['matrix']
            bcs = np.array([b.decode() for b in m['barcodes'][:]]); gn = [t.decode() for t in m['features']['name'][:]]
            M = sp.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=tuple(m['shape'][:]))
        pos = pd.read_parquet(f'{b2}/spatial/tissue_positions.parquet').set_index('barcode').loc[bcs]
        by = (pos['pxl_row_in_fullres'].values * mpp / UM).astype(np.float32); bx = (pos['pxl_col_in_fullres'].values * mpp / UM).astype(np.float32)
        Cm = M.tocoo(); cnt = Cm.data.astype(np.int64)
        rb = np.repeat(Cm.col.astype(np.int64), cnt); rg = np.repeat(Cm.row.astype(np.int64), cnt)
        rng = np.random.default_rng(1); half = np.float32(1.0 / UM)
        yy = by[rb] + rng.uniform(-half, half, len(rb)).astype(np.float32)
        xx = bx[rb] + rng.uniform(-half, half, len(rb)).astype(np.float32)
        save(cd, yy, xx, rg, gn, H, W)
    else:
        log('skip', case)
