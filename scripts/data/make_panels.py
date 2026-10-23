"""Group benchmark cases by gene panel for panel-matched training (TxCellNet and every cell-level baseline).

A group's gene list = model vocabulary (kidney union, 495 genes) intersected with the panel of every member case,
so a model trained on it never reads a gene that one of its target datasets does not measure. Only gene lists are
used (no target labels). Panels covering >= 95% of the vocabulary go to the unrestricted 'full' group.
Existing cases: counts_genes.json; new Xenium cases (not yet prepared): features of cell_feature_matrix.h5.
Also writes the group 'kidney_shared' (no cases): the genes measured in both kidney sections, read by the single-section
kidney networks of the other-section setting (train_gen.py PANEL=kidney_shared).
usage: make_panels.py  -> results/bench/panels.json
"""
import json, os, glob
import h5py, numpy as np

R, C = 'data/xenium_protein_rcc/pilot_L2', 'data/xenium_protein_ccrcc/pilot_L2'
UNION = sorted(set(json.load(open(f'{R}/meta.json'))['genes']) | set(json.load(open(f'{C}/meta.json'))['genes']))
NEW = {'xen_breast_5k': 'pre_breast_5k', 'xen_cervix_5k': 'pre_cervix_5k', 'xen_ovary_5k': 'pre_ovary_5k',
       'xen_prostate_5k': 'pre_prostate_5k', 'xen_skin_5k': 'pre_skin_5k', 'xen_gbm_v1': 'pre_gbm_v1',
       'xen_ovary_v1ff': 'pre_ovary_v1_ff'}
GROUPS = {
    'full': ['kidney_rcc', 'kidney_ccrcc', 'vhd_lung_ffpe', 'vhd_breast_ffpe', 'vhd_tonsil_ff'],
    'v1_377': ['xen_lung_v1', 'xen_pancreas_v1'],
    'breast_280': [f'xen_breast_S{s}_{p}' for s in (1, 2, 3, 4) for p in ('Top', 'Mid', 'Bot')],
    'crlm_462': ['crlm_tumor1', 'crlm_tumor2', 'crlm_adjnormal'],
    'crlm_319': ['crlm_tumor3_A1', 'crlm_tumor3_A2', 'crlm_tumor3_A3'],
    'x5k': ['duchini_core01_hcc', 'duchini_core02_non_tumour', 'duchini_core03_tonsil', 'duchini_core04_hca', 'xen_lymphnode_5k',
            'xen_breast_5k', 'xen_cervix_5k', 'xen_ovary_5k', 'xen_prostate_5k', 'xen_skin_5k'],
    'gbm_v1': ['xen_gbm_v1'],
    'ovary_v1ff': ['xen_ovary_v1ff'],
}


def panel(case):
    f = f'data/cases/{case}/counts_genes.json'
    if os.path.exists(f): return set(json.load(open(f)))
    with h5py.File(f'data/pre/{NEW[case]}/cell_feature_matrix.h5', 'r') as h:
        ft = np.array([t.decode() for t in h['matrix/features/feature_type'][:]]); nm = np.array([t.decode() for t in h['matrix/features/name'][:]])
    return set(nm[ft == 'Gene Expression'])


out = {}
for gname, cases in GROUPS.items():
    per = {c: set(UNION) & panel(c) for c in cases}
    genes = sorted(set(UNION).intersection(*per.values())) if gname != 'full' else UNION
    out[gname] = dict(cases=cases, genes=genes)
    print(f'{gname:11s} {len(genes):3d} genes; per case:', {c: len(v) for c, v in per.items()})
both = sorted(set(json.load(open(f'{R}/meta.json'))['genes']) & set(json.load(open(f'{C}/meta.json'))['genes']))
out['kidney_shared'] = dict(cases=[], genes=both)
print(f'kidney_shared {len(both):3d} genes (measured in both kidney sections)')
os.makedirs('results/bench', exist_ok=True)
json.dump(out, open('results/bench/panels.json', 'w'), indent=1)
print('saved results/bench/panels.json')
