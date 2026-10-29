Data
====

All datasets of the paper are public. No data, results or model weights are stored in the repository. This page lists
where to download the data and where the scripts expect them (paths relative to the repository root);
`data/README.md <https://github.com/wrab12/TxCellNet/blob/main/data/README.md>`_ in the repository holds the same
information and the labels of every benchmark section.

Kidney data of the tutorials
----------------------------

The quick-start tutorial needs no data. The kidney tutorial needs the two 10x Genomics Xenium Protein kidney sections,
which are also the training data of the paper:

* RCC: ``Xenium_V1_Human_Kidney_FFPE_Protein_updated`` (4.0.0;
  `dataset page <https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-carcinoma>`__), unpacked into
  ``data/xenium_protein_rcc/outs/``
* ccRCC: ``Xenium_V1_Human_Clear_Cell_Renal_Cell_Carcinoma_FFPE_Protein`` (4.0.0;
  `dataset page <https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-ccrcc>`__), unpacked into
  ``data/xenium_protein_ccrcc/outs/``

The preparation scripts read ``morphology_focus/``, ``transcripts.parquet``, ``cells.parquet``,
``cell_boundaries.parquet``, ``nucleus_boundaries.parquet``, ``cell_feature_matrix.h5`` and ``gene_panel.json`` of each
bundle.

Sources
-------

10x Genomics datasets are given by bundle name and version. Xenium bundles are under
``https://cf.10xgenomics.com/samples/xenium/<version>/<bundle>/`` and Visium HD bundles under
``https://cf.10xgenomics.com/samples/spatial-exp/<version>/<bundle>/``; each is also listed on the
`10x Genomics datasets page <https://www.10xgenomics.com/datasets>`_.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dataset (sections)
     - Source
   * - Kidney cancer, training (2)
     - RCC: ``Xenium_V1_Human_Kidney_FFPE_Protein_updated`` (4.0.0); ccRCC:
       ``Xenium_V1_Human_Clear_Cell_Renal_Cell_Carcinoma_FFPE_Protein`` (4.0.0)
   * - Lung cancer (1)
     - ``Xenium_V1_humanLung_Cancer_FFPE`` (2.0.0)
   * - Pancreas (1)
     - ``Xenium_V1_human_Pancreas_FFPE`` (2.0.0)
   * - Breast cancer (12)
     - ``Human_Breast_Biomarkers_S{1..4}_{Top,Mid,Bot}`` (4.0.0): patients S1-S4, regions Top, Mid, Bot
   * - Reactive lymph node (1)
     - ``Xenium_Prime_Human_Lymph_Node_Reactive_FFPE`` (3.0.0)
   * - Colorectal liver metastasis (6)
     - Xenium: GEO `GSE335552 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE335552>`_; IMC: figshare
       `32671185 <https://figshare.com/articles/dataset/32671185>`_; Eapen et al. 2026
   * - Liver and tonsil cores (4)
     - Xenium 5K and COMET: Zenodo `21713660 <https://doi.org/10.5281/zenodo.21713660>`_; Duchini et al. 2026
   * - Melanoma (1)
     - ``Xenium_Prime_Human_Skin_FFPE`` (3.0.0)
   * - Prostate cancer (1)
     - ``Xenium_Prime_Human_Prostate_FFPE`` (3.0.0)
   * - Ovarian cancer (1)
     - ``Xenium_Prime_Ovarian_Cancer_FFPE_XRrun`` (3.0.0)
   * - Breast cancer, 5K (1)
     - ``Xenium_Prime_Breast_Cancer_FFPE`` (3.0.0)
   * - Cervical cancer (1)
     - ``Xenium_Prime_Cervical_Cancer_FFPE`` (3.0.0)
   * - Ovarian cancer, fresh frozen (1)
     - ``Xenium_V1_Human_Ovary_Cancer_FF`` (4.0.0)
   * - Glioblastoma (1)
     - ``Xenium_V1_Human_Brain_GBM_FFPE`` (2.0.0)
   * - Visium HD stress test (3; Supplementary Section S4)
     - ``Visium_HD_Human_Lung_Cancer`` (3.0.0); ``Visium_HD_Human_Breast_Cancer_FFPE`` (3.1.2);
       ``Visium_HD_Human_Tonsil_Fresh_Frozen_IF`` (3.1.1)

Gene embeddings. The networks trained on a single non-kidney section read every real panel gene that has an entry in
the GenePT gene embedding table (``GenePT_gene_embedding_ada_text.pickle``, from the data release linked in the
`GenePT repository <https://github.com/yiqunchen/GenePT>`_; only the gene names are used). Download it only from that
release: the file is a Python pickle.

H&E (Supplementary Section S5 only). The post-Xenium H&E image and its alignment file of each 10x Xenium bundle that
provides them (``<bundle>_he_image.ome.tif``, ``<bundle>_he_imagealignment.csv``; the two kidney sections, lung,
pancreas, reactive lymph node and the 12 breast sections). Copy them into one folder per section,
``data/he/<NAME>/he_image.ome.tif`` and ``data/he/<NAME>/he_imagealignment.csv`` (``NAME`` as in
`data/README.md <https://github.com/wrab12/TxCellNet/blob/main/data/README.md>`_); this folder is the ``HE_DIR`` of
``scripts/baselines/he/prep_he.py``.

Expected layout
---------------

.. code-block:: text

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

Folders of the 10x bundles under ``data/ext`` and ``data/pre``, and the benchmark section (``NAME``, the directory under
``data/cases``) prepared from each:

.. list-table::
   :header-rows: 1

   * - Folder
     - 10x bundle
     - NAME
   * - ``ext/xen_lung_v1``
     - ``Xenium_V1_humanLung_Cancer_FFPE``
     - ``xen_lung_v1``
   * - ``ext/xen_pancreas_v1``
     - ``Xenium_V1_human_Pancreas_FFPE``
     - ``xen_pancreas_v1``
   * - ``ext/xen_breast_S{k}_{r}``
     - ``Human_Breast_Biomarkers_S{k}_{r}``
     - ``xen_breast_S{k}_{r}``
   * - ``ext/xen_lymphnode_5k``
     - ``Xenium_Prime_Human_Lymph_Node_Reactive_FFPE``
     - ``xen_lymphnode_5k``
   * - ``pre/pre_skin_5k``
     - ``Xenium_Prime_Human_Skin_FFPE``
     - ``xen_skin_5k``
   * - ``pre/pre_prostate_5k``
     - ``Xenium_Prime_Human_Prostate_FFPE``
     - ``xen_prostate_5k``
   * - ``pre/pre_ovary_5k``
     - ``Xenium_Prime_Ovarian_Cancer_FFPE_XRrun``
     - ``xen_ovary_5k``
   * - ``pre/pre_breast_5k``
     - ``Xenium_Prime_Breast_Cancer_FFPE``
     - ``xen_breast_5k``
   * - ``pre/pre_cervix_5k``
     - ``Xenium_Prime_Cervical_Cancer_FFPE``
     - ``xen_cervix_5k``
   * - ``pre/pre_ovary_v1_ff``
     - ``Xenium_V1_Human_Ovary_Cancer_FF``
     - ``xen_ovary_v1ff``
   * - ``pre/pre_gbm_v1``
     - ``Xenium_V1_Human_Brain_GBM_FFPE``
     - ``xen_gbm_v1``
   * - ``ext/vhd_lung_ffpe``
     - ``Visium_HD_Human_Lung_Cancer``
     - ``vhd_lung_ffpe``
   * - ``ext/vhd_breast_ffpe``
     - ``Visium_HD_Human_Breast_Cancer_FFPE``
     - ``vhd_breast_ffpe``
   * - ``ext/vhd_tonsil_ff``
     - ``Visium_HD_Human_Tonsil_Fresh_Frozen_IF``
     - ``vhd_tonsil_ff``

A Visium HD folder holds the bundle's unpacked ``binned_outputs/`` (the ``square_002um`` bins are read) and its
full-resolution microscope image, saved as ``tissue_image.tif`` or ``tissue_image.btf``. The CRLM sections and the liver
and tonsil cores are prepared from the GEO, figshare and Zenodo files by ``scripts/data/prep_case_crlm.py`` and
``scripts/data/prep_case_duchini.py``; their section names, labels and IMC regions of interest are listed in
`data/README.md <https://github.com/wrab12/TxCellNet/blob/main/data/README.md>`_.

Prepared sections
-----------------

Each benchmark section is prepared into one directory under ``data/cases`` on a 0.85 µm/px grid
(``txcellnet.load_case_image`` reads it):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - File
     - Content
   * - ``dapi.npy``
     - float16, H x W: log1p(DAPI), z-scored over tissue pixels
   * - ``lab.npy``
     - int32, H x W: cell labels (0 = background); label k is the row of ``cells.parquet`` with ``label == k``
   * - ``tx.npz``
     - ``gy``, ``gx`` (int32 grid coordinates), ``gidx`` (index into ``genes.json``), one row per molecule
   * - ``genes.json``
     - genes of the section's panel that belong to the kidney training vocabulary
   * - ``cells.parquet``
     - ``label``, ``cell_id``, ``area_px``, ``x_um``, ``y_um``, ``truth_<marker>`` (raw per-cell mean intensity), ...
   * - ``counts.npz``
     - CSR cell x gene counts (rows ordered like ``cells.parquet``, columns like ``counts_genes.json``)
   * - ``meta.json``
     - dataset, platform, tissue, markers {name: {truth, components (kidney proteins), own (coding genes)}}
