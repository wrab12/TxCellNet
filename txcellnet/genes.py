"""Gene tables: control features, GenePT symbols (gene table of the networks trained on a single non-kidney section) and
the protein -> coding-gene map used for the cognate-mRNA references."""
import os
import pickle

import numpy as np

# GenePT text embeddings (Chen & Zou), used only to decide which panel symbols are real genes with an entry. Download the
# file from the official GenePT data release; the folder name 'GenePT_emebdding_v2' is spelled as in that release.
GENEPT = os.environ.get('TXCELLNET_GENEPT', 'data/genept/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle')
ALIAS = {'CD45RA': 'PTPRC', 'CD45RO': 'PTPRC'}
CONTROL_PREFIXES = ('NegControl', 'BLANK', 'Unassigned', 'Deprecated', 'Intergenic', 'antisense')
_GP = None


def genept(path=None):
    """symbol (upper case) -> GenePT vector"""
    global _GP
    if _GP is None:
        with open(path or GENEPT, 'rb') as f:
            _GP = {str(k).upper(): np.asarray(v, np.float32) for k, v in pickle.load(f).items()}
    return _GP


# protein short name -> coding genes (summed); 'KRT*' = prefix; '|' = fallback if none found
MAP = {
    'PD-1': ['PDCD1'], 'VISTA': ['VSIR'], 'PD-L1': ['CD274'], 'LAG-3': ['LAG3'],
    'CD4': ['CD4'], 'CD20': ['MS4A1'], 'CD8A': ['CD8A'], 'CD3E': ['CD3E'],
    'CD138': ['SDC1'], 'HLA-DR': ['HLA-DRA', 'HLA-DRB1', 'HLA-DRB5'], 'CD11c': ['ITGAX'],
    'CD68': ['CD68'], 'CD16': ['FCGR3A', 'FCGR3B'], 'GranzymeB': ['GZMB'], 'CD163': ['CD163'],
    'CD45RA': ['CD45RA', '|', 'PTPRC'], 'CD45RO': ['CD45RO', '|', 'PTPRC'], 'PCNA': ['PCNA'], 'Ki-67': ['MKI67'],
    'beta-catenin': ['CTNNB1'], 'CD31': ['PECAM1'], 'PTEN': ['PTEN'], 'PanCK': ['KRT*'],
    'Vimentin': ['VIM'], 'alphaSMA': ['ACTA2'], 'CD45': ['PTPRC'], 'E-cadherin': ['CDH1'],
}


def own_cols(genes, marker):
    """indices into `genes` of the coding genes of protein `marker` (case-insensitive name; first non-empty alternative)"""
    alts = ' '.join({k.lower(): v for k, v in MAP.items()}.get(marker.lower(), [])).split('|')
    for alt in alts:
        pats = alt.split()
        cols = [i for i, gname in enumerate(genes)
                if any(gname == q or (q.endswith('*') and gname.startswith(q[:-1])) for q in pats)]
        if cols: return cols
    return []
