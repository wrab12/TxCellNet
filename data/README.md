# Data

No data, results or model weights are stored in this repository. All datasets used in the paper are public; this page lists
where to download them and where the scripts expect them (paths relative to the repository root).

## Sources

10x Genomics datasets are given by bundle name and version. Xenium bundles are under
`https://cf.10xgenomics.com/samples/xenium/<version>/<bundle>/` and Visium HD bundles under
`https://cf.10xgenomics.com/samples/spatial-exp/<version>/<bundle>/`; each is also listed on the
[10x Genomics datasets page](https://www.10xgenomics.com/datasets).

| Dataset (sections) | Source |
|---|---|
| Kidney cancer, training (2) | RCC: `Xenium_V1_Human_Kidney_FFPE_Protein_updated` (4.0.0; [dataset page](https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-carcinoma)); ccRCC: `Xenium_V1_Human_Clear_Cell_Renal_Cell_Carcinoma_FFPE_Protein` (4.0.0; [dataset page](https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-ccrcc)) |
| Lung cancer (1) | `Xenium_V1_humanLung_Cancer_FFPE` (2.0.0) |
| Pancreas (1) | `Xenium_V1_human_Pancreas_FFPE` (2.0.0) |
| Breast cancer (12) | `Human_Breast_Biomarkers_S{1..4}_{Top,Mid,Bot}` (4.0.0): patients S1-S4, regions Top, Mid, Bot |
| Reactive lymph node (1) | `Xenium_Prime_Human_Lymph_Node_Reactive_FFPE` (3.0.0) |
| Colorectal liver metastasis (6) | Xenium: GEO [GSE335552](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE335552); IMC: figshare [32671185](https://figshare.com/articles/dataset/32671185); Eapen et al. 2026 |
| Liver and tonsil cores (4) | Xenium 5K and COMET: Zenodo [21713660](https://doi.org/10.5281/zenodo.21713660); Duchini et al. 2026 |
| Melanoma (1) | `Xenium_Prime_Human_Skin_FFPE` (3.0.0) |
| Prostate cancer (1) | `Xenium_Prime_Human_Prostate_FFPE` (3.0.0) |
| Ovarian cancer (1) | `Xenium_Prime_Ovarian_Cancer_FFPE_XRrun` (3.0.0) |
| Breast cancer, 5K (1) | `Xenium_Prime_Breast_Cancer_FFPE` (3.0.0) |
| Cervical cancer (1) | `Xenium_Prime_Cervical_Cancer_FFPE` (3.0.0) |
| Ovarian cancer, fresh frozen (1) | `Xenium_V1_Human_Ovary_Cancer_FF` (4.0.0) |
| Glioblastoma (1) | `Xenium_V1_Human_Brain_GBM_FFPE` (2.0.0) |
| Visium HD stress test (3; Supplementary Section S4) | `Visium_HD_Human_Lung_Cancer` (3.0.0); `Visium_HD_Human_Breast_Cancer_FFPE` (3.1.2); `Visium_HD_Human_Tonsil_Fresh_Frozen_IF` (3.1.1) |

Gene embeddings: the networks trained on a single non-kidney section read every real panel gene that has an entry in the
GenePT gene embedding table (`GenePT_gene_embedding_ada_text.pickle`, from the data release linked in the
[GenePT repository](https://github.com/yiqunchen/GenePT); only the gene names are used). Download it only from that release:
the file is a Python pickle.

H&E (Supplementary Section S5 only): the post-Xenium H&E image and its alignment file of each 10x Xenium bundle that
provides them (`<bundle>_he_image.ome.tif`, `<bundle>_he_imagealignment.csv`; the two kidney sections, lung, pancreas,
reactive lymph node and the 12 breast sections). Copy them into one folder per section, `data/he/<NAME>/he_image.ome.tif`
and `data/he/<NAME>/he_imagealignment.csv` (`NAME` as in the tables below); this folder is the `HE_DIR` of
`scripts/baselines/he/prep_he.py`.

## Expected layout

```
data/
├── xenium_protein_rcc/outs/          # kidney RCC bundle (morphology_focus/, transcripts.parquet, cells.parquet,
├── xenium_protein_ccrcc/outs/        #   cell_boundaries.parquet, nucleus_boundaries.parquet, cell_feature_matrix.h5, gene_panel.json)
├── xenium_protein_<s>/pilot_L2/      # training grid written by scripts/data/prep_kidney_grid.py; runs/<tag>/ holds cfg.json and
│                                     #   model.pt of each network
├── ext/<folder>/                     # other Xenium bundles (contents of outs/) and the Visium HD bundles; folders below
├── pre/<folder>/                     # the seven Xenium sections added after the configuration freeze (contents of outs/); folders below
├── crlm_xenium_imc/
│   ├── xenium/ex/                    # GEO files <sample>_{morphology.ome.tif, transcripts.parquet, cell_boundaries.parquet,
│   │                                 #   cell_feature_matrix.h5}; samples GSM9815454_tumor1_M242T, GSM9815455_tumor2_M131T,
│   │                                 #   GSM9815456_tumor3_M397T, GSM9815457_AdjNormal_M131N
│   ├── mcd/                          # figshare .mcd files: tumor1_M242T, tumor2_M131T, AdjNormal_M131N, postxnmIMC_tumor3_M397T
│   └── imc_npz/                      # written by scripts/data/extract_imc.py
├── duchini_xenium_comet/ex/          # Zenodo archive, unpacked: UnumLocalia/<core>/{xenium,comet}/, COMET_ome_tiff/, Cell_quantifications/
├── genept/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle   # folder name as in the GenePT release
├── he/<NAME>/                        # H&E baseline only: he_image.ome.tif, he_imagealignment.csv (see Sources);
│                                     #   he/kidney_{rcc,ccrcc}/he_grid.npy is written by scripts/baselines/he/prep_he.py
├── cases/<NAME>/                     # benchmark sections, written by scripts/data/prep_case_*.py
└── sim/                              # semi-synthetic benchmark, written by scripts/data/sim_make.py
```

Folders of the 10x bundles under `data/ext` and `data/pre`, and the benchmark section (`NAME`) prepared from each:

| Folder | 10x bundle | NAME |
|---|---|---|
| `ext/xen_lung_v1` | `Xenium_V1_humanLung_Cancer_FFPE` | `xen_lung_v1` |
| `ext/xen_pancreas_v1` | `Xenium_V1_human_Pancreas_FFPE` | `xen_pancreas_v1` |
| `ext/xen_breast_S{k}_{r}` | `Human_Breast_Biomarkers_S{k}_{r}` | `xen_breast_S{k}_{r}` |
| `ext/xen_lymphnode_5k` | `Xenium_Prime_Human_Lymph_Node_Reactive_FFPE` | `xen_lymphnode_5k` |
| `pre/pre_skin_5k` | `Xenium_Prime_Human_Skin_FFPE` | `xen_skin_5k` |
| `pre/pre_prostate_5k` | `Xenium_Prime_Human_Prostate_FFPE` | `xen_prostate_5k` |
| `pre/pre_ovary_5k` | `Xenium_Prime_Ovarian_Cancer_FFPE_XRrun` | `xen_ovary_5k` |
| `pre/pre_breast_5k` | `Xenium_Prime_Breast_Cancer_FFPE` | `xen_breast_5k` |
| `pre/pre_cervix_5k` | `Xenium_Prime_Cervical_Cancer_FFPE` | `xen_cervix_5k` |
| `pre/pre_ovary_v1_ff` | `Xenium_V1_Human_Ovary_Cancer_FF` | `xen_ovary_v1ff` |
| `pre/pre_gbm_v1` | `Xenium_V1_Human_Brain_GBM_FFPE` | `xen_gbm_v1` |
| `ext/vhd_lung_ffpe` | `Visium_HD_Human_Lung_Cancer` | `vhd_lung_ffpe` |
| `ext/vhd_breast_ffpe` | `Visium_HD_Human_Breast_Cancer_FFPE` | `vhd_breast_ffpe` |
| `ext/vhd_tonsil_ff` | `Visium_HD_Human_Tonsil_Fresh_Frozen_IF` | `vhd_tonsil_ff` |

A Visium HD folder holds the bundle's unpacked `binned_outputs/` (the `square_002um` bins are read) and its full-resolution
microscope image, saved as `tissue_image.tif` or `tissue_image.btf`.

## Benchmark sections

`NAME` is the directory under `data/cases/`. `TISSUE` and `PANEL` are labels copied into Supplementary Table S11
(`bench_cases.csv`); they do not affect any computation.

| Section(s) | NAME | Preparation (input) | TISSUE | PANEL |
|---|---|---|---|---|
| Kidney RCC | `kidney_rcc` | `prep_case_xenium.py` (`data/xenium_protein_rcc/outs`) | `kidney_RCC` | `v1_377+28` |
| Kidney ccRCC | `kidney_ccrcc` | `prep_case_xenium.py` (`data/xenium_protein_ccrcc/outs`) | `kidney_ccRCC` | `v1_377+100` |
| Lung cancer | `xen_lung_v1` | `prep_case_xenium.py` (`data/ext/NAME`) | `lung_cancer` | `v1_multitissue` |
| Pancreas | `xen_pancreas_v1` | `prep_case_xenium.py` (`data/ext/NAME`) | `pancreas` | `v1_multitissue` |
| Breast cancer | `xen_breast_S{1..4}_{Top,Mid,Bot}` | `prep_case_xenium.py` (`data/ext/NAME`) | `breast` (`xen_breast_S1_Top`: `breast_IDC`) | `custom280` |
| Reactive lymph node | `xen_lymphnode_5k` | `prep_case_xenium.py` (`data/ext/NAME`) | `lymph_node_reactive` | `prime5k` |
| Melanoma | `xen_skin_5k` | `prep_case_xenium.py` (`data/pre/pre_skin_5k`) | `skin melanoma` | `Xenium 5K Pan Tissue & Pathways` |
| Prostate cancer | `xen_prostate_5k` | `prep_case_xenium.py` (`data/pre/pre_prostate_5k`) | `prostate adenocarcinoma` | `Xenium 5K Pan Tissue & Pathways` |
| Ovarian cancer | `xen_ovary_5k` | `prep_case_xenium.py` (`data/pre/pre_ovary_5k`) | `ovarian cancer` | `Xenium 5K Cell Typing + add-on` |
| Breast cancer, 5K | `xen_breast_5k` | `prep_case_xenium.py` (`data/pre/pre_breast_5k`) | `breast cancer` | `Xenium 5K Cell Typing + add-on` |
| Cervical cancer | `xen_cervix_5k` | `prep_case_xenium.py` (`data/pre/pre_cervix_5k`) | `cervical cancer` | `Xenium 5K Cell Typing + add-on` |
| Ovarian cancer, fresh frozen | `xen_ovary_v1ff` | `prep_case_xenium.py` (`data/pre/pre_ovary_v1_ff`) | `ovarian cancer (fresh frozen)` | `Xenium Multi-Tissue and Cancer + 100 custom` |
| Glioblastoma | `xen_gbm_v1` | `prep_case_xenium.py` (`data/pre/pre_gbm_v1`) | `glioblastoma` | `Xenium Immuno-Oncology add-on B` |
| CRLM | `crlm_tumor1`, `crlm_tumor2`, `crlm_adjnormal`, `crlm_tumor3_A1`, `crlm_tumor3_A2`, `crlm_tumor3_A3` | `prep_case_crlm.py` (Xenium sample + IMC ROI) | set by the script | set by the script |
| Liver and tonsil cores | `duchini_core01_hcc`, `duchini_core02_non_tumour`, `duchini_core03_tonsil`, `duchini_core04_hca` | `prep_case_duchini.py` (`CORE` = `core01_hcc`, ...) | set by the script | set by the script |

Visium HD sections (`prep_case_visiumhd.py data/ext/NAME data/cases/NAME NAME TISSUE CHANNEL_MAP_JSON`; the channel map gives
the channels of the IF image):

| NAME | TISSUE | CHANNEL_MAP_JSON |
|---|---|---|
| `vhd_lung_ffpe` | `lung_cancer` | `'{"0": "CD45RO", "1": "CD20", "2": "DAPI"}'` |
| `vhd_breast_ffpe` | `breast_DCIS` | `'{"0": "CD45RO", "1": "CD20", "2": "DAPI"}'` |
| `vhd_tonsil_ff` | `tonsil_FF` | `'{"0": "CD45", "1": "aSMA", "2": "CD20", "3": "DAPI"}'` |

IMC regions of interest of the CRLM sections (`data/crlm_xenium_imc/imc_npz/<ROI>.npz`):

| NAME | Xenium sample | ROI |
|---|---|---|
| `crlm_tumor1` | `GSM9815454_tumor1_M242T` | `tumor1_M242T__Acquisition_ROI_1` |
| `crlm_tumor2` | `GSM9815455_tumor2_M131T` | `tumor2_M131T__Acquisition_ROI_1` |
| `crlm_adjnormal` | `GSM9815457_AdjNormal_M131N` | `AdjNormal_M131N__Acquisition_ROI_1` |
| `crlm_tumor3_A1` | `GSM9815456_tumor3_M397T` | `postxnmIMC_tumor3_M397T__ROI_002-A1` |
| `crlm_tumor3_A2` | `GSM9815456_tumor3_M397T` | `postxnmIMC_tumor3_M397T__ROI_003-A2` |
| `crlm_tumor3_A3` | `GSM9815456_tumor3_M397T` | `postxnmIMC_tumor3_M397T__ROI_008-A3` |
