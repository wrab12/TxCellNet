"""Per-protein cell-level Spearman of each method on two datasets (main Fig. 2b and Supplementary Table S13).
kidney  held-out stripes of the two kidney sections, same-section setting: the models of Supplementary Table S4 (TxCellNet
        one clean network, NicheTrans and MLP single models); mean of the two sections; 27 proteins (the two stains left out).
crlm    the four CRLM test sections (IMC), kidney-trained: the models of Supplementary Table S3 (six-network TxCellNet
        ensemble, NicheTrans and MLP three-seed ensembles); median of the four sections, as in Supplementary Fig. S2; 22 proteins.
Input: DATA_DIR/per_protein/bench_rows_kidney_crlm.csv (rows of results/bench/*/bench_rows.csv for the kidney and CRLM
sections) and DATA_DIR/tables/fig_{within,cross}_per_dataset.csv (per-section medians behind Supplementary Tables S4 / S3).
Every per-section median over markers is checked against those tables.
"""
import pandas as pd

METHODS = ['txcellnet', 'nichetrans', 'dgat', 'spamosaic', 'totalvi', 'mlp', 'own_mRNA']
# same-section setting on kidney = the kidney rows of Supplementary Table S4; kidney-trained CRLM = Supplementary Table S3
KMAP = {'txcellnet': 'fin_clean_s1', 'nichetrans': 'nichetrans_off_u', 'dgat': 'dgat_pm', 'spamosaic': 'spamosaic_pm',
        'totalvi': 'totalvi_pm', 'mlp': 'mlp_pm_u', 'own_mRNA': 'own_mRNA'}
XMAP = {'txcellnet': 'txcellnet', 'nichetrans': 'nichetrans_off_e3', 'dgat': 'dgat_pm', 'spamosaic': 'spamosaic_pm',
        'totalvi': 'totalvi_pm', 'mlp': 'mlp_pm_e3', 'own_mRNA': 'own_mRNA'}
KID, CRLM_T = ['kidney_rcc', 'kidney_ccrcc'], ['crlm_tumor2', 'crlm_tumor3_A2', 'crlm_tumor3_A3', 'crlm_adjnormal']
STAINS = ['interior_aSMA_Vim', 'boundary_ATP1A1_Ecad_CD45']
# protein categories, in the order of Supplementary Fig. S2
CAT = [('Immune lineage', ['CD20', 'CD3E', 'CD4', 'CD8A', 'CD45', 'CD45RO', 'CD45RA', 'CD68', 'CD163', 'CD11c', 'CD16', 'CD138']),
       ('Epithelial, tumor', ['PanCK', 'K8_18', 'E-Cadherin', 'Beta-catenin']),
       ('Stromal, vascular', ['Vimentin', 'alphaSMA', 'CD31']),
       ('Cell state, checkpoint', ['Ki-67', 'PCNA', 'PD-1', 'PD-L1', 'LAG-3', 'VISTA', 'GranzymeB', 'HLA-DR', 'PTEN'])]
PRETTY = {'E-Cadherin': 'E-cadherin', 'Beta-catenin': 'β-catenin', 'alphaSMA': 'αSMA', 'GranzymeB': 'Granzyme B', 'CD8A': 'CD8',
          'CD3E': 'CD3', 'K8_18': 'K8/18'}


def per_protein(dd):
    """(PK, PCR): proteins x METHODS, reindexed to the 28 proteins of CAT (NaN = not measured / no cognate transcript)"""
    R = pd.read_csv(f'{dd}/per_protein/bench_rows_kidney_crlm.csv')
    T4 = pd.read_csv(f'{dd}/tables/fig_within_per_dataset.csv', index_col=0)
    T3 = pd.read_csv(f'{dd}/tables/fig_cross_per_dataset.csv', index_col=0)
    # the rows reproduce the per-section medians of Supplementary Tables S4 / S3 (all markers incl. the two stains)
    for cases, MAP, T in ((KID, KMAP, T4), (CRLM_T, XMAP, T3)):
        for c in cases:
            for m, k in MAP.items():
                assert abs(R[(R.case == c) & (R.method == k)].spearman.median() - T.loc[c, m]) < 1e-9, (c, m)
    Rk = R[R.case.isin(KID) & ~R.marker.isin(STAINS)]
    Rc = R[R.case.isin(CRLM_T)]
    PK = pd.DataFrame({m: Rk[Rk.method == k].groupby('marker').spearman.mean() for m, k in KMAP.items()})       # mean of 2 sections
    PCR = pd.DataFrame({m: Rc[Rc.method == k].groupby('marker').spearman.median() for m, k in XMAP.items()})    # median of 4 sections
    assert len(PK) == 27 and len(PCR) == 22
    prots = [p for _, c in CAT for p in c]
    assert set(prots) == set(PK.index) | set(PCR.index) and len(prots) == 28
    PK, PCR = PK.reindex(prots), PCR.reindex(prots)
    PK.index.name = PCR.index.name = 'marker'
    return PK, PCR
