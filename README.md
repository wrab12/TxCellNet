# TxCellNet: transferable generation of multiplexed protein images from spatial transcriptomics

## Introduction
TxCellNet generates multiplexed protein images from imaging-based spatial transcriptomics. It reads what every
imaging-based section provides: the position and gene of each transcript molecule, the DAPI image and the cell
segmentation. Each molecule is replaced by a learnable embedding of its gene, and the embeddings are summed at the
molecule's pixel. A gene that a panel does not measure contributes nothing to this gene field, so one model reads any gene
panel (the panel genes in its training vocabulary). A U-Net reads the gene field together with DAPI, and a
cell-neighbourhood transformer pools the U-Net features and the gene field of each cell into a token and lets every cell
attend to the other cells of the window, with a Gaussian distance bias whose length scale each attention head learns.
Stage 1 predicts the conditional-mean protein image; its mean over the pixels of a cell is that cell's protein prediction.
To transfer to other tissues, gene panels, detection depths and imaging conditions, TxCellNet averages networks trained on
clean data and networks trained on randomized data (molecule thinning, gene dropout and DAPI perturbation). Stage 2 is a
rectified-flow model conditioned on the stage-1 image, DAPI and cell geometry. It adds the sub-cellular texture of
measured images through a residual whose mean is removed in every cell, so each cell keeps its stage-1 value. Cell values
are taken from the stage-1 image; the refined image is for reading and side-by-side comparison with measured images.

We evaluated TxCellNet on 34 sections from 14 Xenium datasets with four kinds of protein readout (same-section
immunofluorescence, multimodal segmentation stains, imaging mass cytometry and sequential immunofluorescence). We compared
it with four published RNA-to-protein methods (NicheTrans, DGAT, SpaMosaic and totalVI) and five baselines in three
settings: trained on the same section, on another section of the same dataset, or only on two kidney cancer sections with
same-section RNA and protein (10x Xenium Protein). All methods are scored on the same cells. TxCellNet is the most accurate
method in all three settings, and its images place membrane and nuclear proteins where the measured images show them,
which per-cell predictions cannot represent.

![TxCellNet overview](TxCellNet.png)

## Repository Structure
```
TxCellNet/
├── txcellnet/                  # Python package (pip install -e .)
│   ├── model.py                # stage 1: gene-embedding splatting, U-Net, cell-neighbourhood transformer (GenNet)
│   ├── flow.py                 # stage 2: flow U-Net, cell geometry channels, sampling with the cell-mean projection
│   ├── predict.py              # whole-section inference (windows with context margins, four flips, per-cell sums)
│   ├── metrics.py              # Spearman, Pearson, AUROC of Otsu-positive cells
│   ├── io.py                   # prepared-section format
│   ├── bench.py                # benchmark data access: kidney metadata, panel groups, splits, scoring, section lists
│   ├── genes.py                # control features, GenePT symbols, protein -> coding-gene map
│   └── utils.py                # device, mixed precision, logging
├── scripts/                    # every step of the paper; run from the repository root (reads data/, writes results/)
│   ├── data/                   # training grids, splits, benchmark sections, image targets, semi-synthetic data
│   ├── train/                  # stage-1 networks, whole-grid kidney prediction, stage 2, single-section networks
│   ├── benchmark/              # scoring of TxCellNet in the three settings, cognate mRNA
│   ├── baselines/              # cell-level baselines, NicheTrans, DGAT, SpaMosaic, totalVI, spot level, H&E (he/)
│   ├── evaluation/             # setting summaries, image-level metrics, robustness, semi-synthetic, case study
│   ├── tables/                 # LaTeX tables of the supplementary material
│   └── figdata/                # image windows and tables read by the figure scripts
├── figures/                    # figure scripts (Figs 2-5, Supplementary Figs S1-S6, Table S13) and make_all.sh
├── data/README.md              # data sources and expected layout (no data in the repository)
├── envs/                       # extra requirements of the PyG (DGAT, SpaMosaic) and scvi (totalVI) environments
├── smoke_test.py               # end-to-end test on synthetic data
├── TxCellNet.png               # overview figure
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Environment
First, clone and navigate to the repository.
```bash
git clone https://github.com/wrab12/TxCellNet
cd TxCellNet
```

Create and activate an environment with Python 3.11 (the version used in the paper) using `conda`.
```bash
conda create -n txcellnet python=3.11
conda activate txcellnet
```

Install PyTorch for your CUDA version ([PyTorch installation guide](https://pytorch.org/get-started/locally/); run
`nvidia-smi` to check the CUDA version). The paper used PyTorch 2.5 with CUDA 12.1:
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the dependencies and the package.
```bash
pip install -r requirements.txt
pip install -e .
```
The package and the smoke test also run on the CPU. Training and whole-section inference expect a CUDA GPU (the paper used
a single NVIDIA A800).

The compared methods run with their official code in two further environments:
```bash
# PyG environment (DGAT, SpaMosaic, spot-level evaluation): the main environment plus torch_geometric 2.6.1
conda create -n txcellnet-pyg --clone txcellnet && conda activate txcellnet-pyg
pip install -r envs/requirements-pyg.txt -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
# scvi environment (totalVI, trained on the CPU): scvi-tools 1.5.1
conda create -n txcellnet-scvi python=3.12 && conda activate txcellnet-scvi
pip install torch==2.8.0 && pip install -r envs/requirements-scvi.txt && pip install -e .
```

Clone the official repositories into `external/` (or set the environment variables to another location):

| Method | Code | Variable (default) |
|---|---|---|
| NicheTrans | https://github.com/YSTLab/NicheTrans (commit 85fab47) | `NICHETRANS_DIR` (`external/NicheTrans`) |
| DGAT | https://github.com/osmanbeyoglulab/DGAT (commit b0128f3) | `DGAT_DIR` (`external/DGAT`) |
| SpaMosaic | https://github.com/JinmiaoChenLab/SpaMosaic (commit cc13367) | `SPAMOSAIC_DIR` (`external/SpaMosaic`) |
| totalVI | scvi-tools 1.5.1 (PyPI) | |

```bash
git clone https://github.com/YSTLab/NicheTrans external/NicheTrans && git -C external/NicheTrans checkout 85fab47
git clone https://github.com/osmanbeyoglulab/DGAT external/DGAT && git -C external/DGAT checkout b0128f3
git clone https://github.com/JinmiaoChenLab/SpaMosaic external/SpaMosaic && git -C external/SpaMosaic checkout cc13367
```

## Testing
Run the smoke test from the repository root. On small synthetic sections it checks the molecule splatting, a few training
steps of stage 1 (gene embeddings and cell-neighbourhood transformer), whole-section inference with genes outside the
vocabulary and with a partial panel, stage-2 sampling, and that every cell mean of the refined image equals the stage-1
cell mean. It takes well under a minute on a laptop CPU.
```bash
python smoke_test.py
```
Expected output (numbers and timings vary with the PyTorch version and hardware):
```
[smoke] torch 2.11.0, device cpu
[smoke] ok    molecule splatting (0.1s)
         stage-1 loss 0.3498 -> 0.0382
[smoke] ok    stage-1 training (5.3s)
[smoke] ok    out-of-vocabulary genes dropped (0.7s)
[smoke] ok    partial panel and panel matching (0.6s)
         stage-2 cell means preserved (max |difference| 9.7e-09)
[smoke] ok    stage 2 keeps every cell mean (0.6s)
[smoke] ok    cell-level metrics (0.0s)
[smoke] ALL PASS (7s)
```
The script exits with a non-zero status if any check fails.

## Datasets
All datasets are public. [data/README.md](data/README.md) lists the download links (10x Genomics, GEO, figshare, Zenodo,
GenePT) and the folder layout the scripts expect. No data are stored in this repository.

## Reproducing the paper
All scripts are run from the repository root, read `data/` and write `results/`; each script documents its inputs, outputs
and options in its docstring. Commands marked `# PyG env` or `# scvi env` run in those environments.
The three settings of the paper and their script modes: same-section (`within`), other-section (`xsec`), cross-dataset
(`cross`; trained on the two kidney sections only). `PANEL_GROUPS` are the panel groups of `scripts/data/make_panels.py`.

```bash
PANEL_GROUPS="full v1_377 breast_280 crlm_462 crlm_319 x5k gbm_v1 ovary_v1ff"
FIN=fin_clean_s1,fin_clean_s2,fin_clean_s3,fin_rand_s1,fin_rand_s2,fin_rand_s3
mkdir -p results/bench results/tables

# 1. kidney training grids (0.85 um/px): DAPI, 27 protein images, cells, transcripts; held-out stripes and vendor per-cell values
for s in rcc ccrcc; do
  python scripts/data/prep_kidney_grid.py xenium_protein_$s
  python scripts/data/prep_kidney_tx.py data/xenium_protein_$s/pilot_L2
done
python scripts/data/make_kidney_splits.py

# 2. benchmark sections, one directory per section under data/cases (names and labels: data/README.md)
python scripts/data/prep_case_xenium.py OUTS data/cases/NAME NAME TISSUE PANEL        # every Xenium section, incl. the kidney ones
python scripts/data/extract_imc.py data/crlm_xenium_imc/imc_npz                       # IMC regions of interest from the .mcd files
python scripts/data/prep_case_crlm.py SAMPLE data/crlm_xenium_imc/imc_npz/ROI.npz data/cases/NAME NAME
python scripts/data/prep_case_duchini.py CORE data/cases/duchini_CORE duchini_CORE
python scripts/data/prep_case_visiumhd.py data/ext/NAME data/cases/NAME NAME TISSUE CHANNEL_MAP_JSON   # Supplementary S4
python scripts/data/add_kidney_eval.py; python scripts/data/add_kidney_protein_truth.py
python scripts/data/fit_cocktail.py results/bench/cocktail_weights.json             # stain-cocktail weights (Table S2)
python scripts/data/make_panels.py                                                    # panel groups -> results/bench/panels.json
EXT=$(ls data/cases | grep -v -e '^kidney' -e '^vhd')          # the 32 external Xenium sections
python scripts/data/add_allgenes.py $EXT       # all panel genes with a GenePT entry (networks trained on one section)
python scripts/data/prep_targets.py $EXT       # image targets on the section grid and the 8-stripe split

# 3. cross-dataset setting (`cross`; trained on the two kidney sections only). The kidney held-out stripes scored here are
#    also the kidney rows of the same-section setting (TxCellNet: fin_clean_s1; NicheTrans and baselines: the
#    GENESET=union runs below; DGAT, SpaMosaic, totalVI: these runs)
#    stage 1: three clean and three randomized networks on the kidney training stripes
for s in 1 2 3; do
  PAUG=0 THIN=1 GDROP=0.5 DAUG=1 SEED=$s python scripts/train/train_gen.py fin_clean_s$s 24000
  PAUG=1 THIN=1 GDROP=0.5 DAUG=1 SEED=$s python scripts/train/train_gen.py fin_rand_s$s 24000
done
#    stage 2 (mean image = the six-network ensemble on the whole kidney grids)
python scripts/train/pred_kidney_full.py $FIN fin_ens6
python scripts/train/train_refine.py fin_ens6 refine_fin
#    TxCellNet on every section (six-network ensemble 'txcellnet'); cognate mRNA and section statistics
ENS="txcellnet=${FIN//,/+}" python scripts/benchmark/run_bench_gen.py results/bench/final $FIN data/cases/*
python scripts/benchmark/own_only.py results/bench/own data/cases/*
#    cell-level baselines, panel-matched (3 seeds, 3-seed MLP ensemble; kidney rows: the 387 genes measured in both kidney sections)
for s in 1 2 3; do SEED=$s python scripts/baselines/baselines_cell.py results/bench/cellbase_s$s all; done
python scripts/baselines/ens_cellbase.py results/bench/cellbase_e3 mlp_pm results/bench/cellbase_s1,results/bench/cellbase_s2,results/bench/cellbase_s3
#    kidney rows of the same-section setting: all 495 vocabulary genes, zero where a section lacks one (seed 1)
GENESET=union python scripts/baselines/baselines_cell.py results/bench/cellbase_u full
#    NicheTrans as in its official Xenium tutorial (8 panel groups x 3 seeds, 3-seed ensemble; kidney rows: 387 shared genes)
for g in $PANEL_GROUPS; do for s in 1 2 3; do SEED=$s python scripts/baselines/nichetrans_official.py cross results/bench/nt_off_${g}_s$s $g; done; done
python scripts/baselines/ens_cellbase.py results/bench/nt_off_e3 nichetrans_off $(ls -d results/bench/nt_off_*_s[123] | paste -sd, -)
GENESET=union python scripts/baselines/nichetrans_official.py cross results/bench/nt_off_full_u full   # same-section kidney rows
python scripts/baselines/nt_graph_equiv.py                                            # CUDA-graph replay = eager execution
#    DGAT (one job per panel group); SpaMosaic and totalVI are transductive (one job per section)
for g in $PANEL_GROUPS; do python scripts/baselines/dgat_bench.py cross results/bench/dgat_$g $g; done          # PyG env
python -c "import json, os; [print(c, g) for g, v in json.load(open('results/bench/panels.json')).items() for c in v['cases']
 if os.path.exists(f'data/cases/{c}/meta.json') and not c.startswith('vhd')]" > results/case_groups.txt
while read c g; do CASES=$c python scripts/baselines/spamosaic_bench.py cross results/bench/spam_$c $g; done < results/case_groups.txt  # PyG env
while read c g; do CASES=$c python scripts/baselines/totalvi_bench.py cross results/bench/tvi_$c $g; done < results/case_groups.txt    # scvi env
#    H&E -> protein with the TxCellNet backbone (Supplementary S5); data/he/NAME holds the H&E of section NAME (data/README.md)
hw() { python -c "import json, sys; m = json.load(open(sys.argv[1])); print(m['H'], m['W'])" "$1"; }   # grid size
for s in rcc ccrcc; do
  python scripts/baselines/he/prep_he.py data/he/kidney_$s data/he/kidney_$s/he_grid.npy $(hw data/xenium_protein_$s/pilot_L2/meta.json)
done
for c in $(ls data/he); do python scripts/baselines/he/prep_he.py data/he/$c data/cases/$c/he.npy $(hw data/cases/$c/meta.json); done
COLAUG=1 SEED=1 python scripts/baselines/he/train_he.py he_aug 12000
python scripts/baselines/he/run_bench_he.py results/bench/he he_aug data/cases/*

# 4. same-section setting (`within`; every external section; the kidney rows are scored in step 3)
for c in $EXT; do
  PAUG=0 SEED=1 python scripts/train/train_case.py $c within_clean 12000      # ACCUM=2 halves the memory (same batch)
  python scripts/baselines/within_baselines.py $c                              # ridge, MLP, kNN, cell-type lookup, cognate mRNA
  python scripts/baselines/nichetrans_official.py within $c
  python scripts/baselines/dgat_bench.py within $c                             # PyG env
  python scripts/baselines/spamosaic_bench.py within $c                        # PyG env
  python scripts/baselines/totalvi_bench.py within $c                          # scvi env
done

# 5. other-section setting (`xsec`; 50 pairs A -> B): breast, same patient and next region (Top -> Mid -> Bot -> Top)
#    and next patient and same region (S1 -> S2 -> S3 -> S4 -> S1); CRLM and COMET, every other section of the same panel
#    group; kidney RCC <-> ccRCC. TxCellNet = A's clean same-section network + a randomized one (averaged).
python - > results/xsec_pairs.txt <<'EOF'
P, R = (1, 2, 3, 4), ('Top', 'Mid', 'Bot')
for p in P:
    for i, r in enumerate(R):
        print(f'xen_breast_S{p}_{r}', f'xen_breast_S{p}_{R[(i + 1) % 3]}'); print(f'xen_breast_S{p}_{r}', f'xen_breast_S{P[p % 4]}_{r}')
for grp in (['crlm_tumor1', 'crlm_tumor2', 'crlm_adjnormal'], ['crlm_tumor3_A1', 'crlm_tumor3_A2', 'crlm_tumor3_A3'],
            ['duchini_core01_hcc', 'duchini_core02_non_tumour', 'duchini_core03_tonsil', 'duchini_core04_hca']):
    for a in grp: print(*[f'{a} {b}' for b in grp if b != a], sep='\n')
EOF
for a in $(cut -d' ' -f1 results/xsec_pairs.txt | sort -u); do
  B=$(awk -v a=$a '$1 == a {print $2}' results/xsec_pairs.txt | paste -sd, -)
  PAUG=1 SEED=1 python scripts/train/train_case.py $a within_rand 12000
  python scripts/benchmark/xsec_txcellnet.py $a $B; TAG=within_rand python scripts/benchmark/xsec_txcellnet.py $a $B
  python scripts/baselines/xsec_baselines.py $a $B                             # ridge, MLP, kNN, cell-type lookup, cognate mRNA
  python scripts/baselines/nichetrans_official.py xsec $a $B
  python scripts/baselines/dgat_bench.py xsec $a $B; python scripts/baselines/spamosaic_bench.py xsec $a $B   # PyG env
  python scripts/baselines/totalvi_bench.py xsec $a $B                                                     # scvi env
done
#    kidney: single-section models reading the genes measured in both sections
for sec in rcc ccrcc; do
  SECTIONS=$sec PANEL=kidney_shared PAUG=0 THIN=1 GDROP=0.5 DAUG=1 SEED=1 python scripts/train/train_gen.py xs_clean_$sec 24000
  SECTIONS=$sec PANEL=kidney_shared PAUG=1 THIN=1 GDROP=0.5 DAUG=1 SEED=1 python scripts/train/train_gen.py xs_rand_$sec 24000
  ENS="xs_cr_$sec=xs_clean_$sec+xs_rand_$sec" python scripts/benchmark/run_bench_gen.py results/bench/xsec_kidney_tx_cr_$sec \
    xs_clean_$sec,xs_rand_$sec data/cases/kidney_rcc data/cases/kidney_ccrcc
  export KID_TRAIN=$sec CASES=kidney_rcc,kidney_ccrcc
  python scripts/baselines/baselines_cell.py results/bench/xsec_kidney/cellbase_$sec full
  python scripts/baselines/nichetrans_official.py cross results/bench/xsec_kidney/nt_$sec full
  python scripts/baselines/dgat_bench.py cross results/bench/xsec_kidney/dgat_$sec full              # PyG env
  python scripts/baselines/spamosaic_bench.py cross results/bench/xsec_kidney/spam_$sec full         # PyG env
  python scripts/baselines/totalvi_bench.py cross results/bench/xsec_kidney/tvi_$sec full            # scvi env
  unset KID_TRAIN CASES
done
python scripts/evaluation/xsec_summary.py results/bench/xsec/summary        # the three settings on identical test cells
python scripts/tables/xsec_table_tex.py results/bench/xsec/summary/xsec_groups.csv results/tables/table_xsec.tex
#    kidney cross-patient column of Supplementary Fig. S2: one mixed single-section network per kidney section
for sec in rcc ccrcc; do
  SECTIONS=$sec PANEL=kidney_shared PAUG=0.5 THIN=1 GDROP=0.5 DAUG=1 SEED=1 python scripts/train/train_gen.py x_${sec}_only 24000
done
python scripts/benchmark/run_bench_gen.py results/bench/cross_kidney x_rcc_only,x_ccrcc_only data/cases/kidney_rcc data/cases/kidney_ccrcc

# 6. spot-level evaluation (Supplementary S9): hexagonal pseudo-spots, PITCH=55 um between neighbouring centres
python scripts/baselines/grid_eval.py within $EXT                            # PyG env
python scripts/baselines/grid_eval.py cross ${PANEL_GROUPS// /,}             # PyG env; group 'full' = kidney held-out stripes
python scripts/evaluation/grid_summary.py results/bench/grid                 # tables, wins, Wilcoxon tests, table_grid_*.tex

# 7. ablations (Supplementary S3): mixed reference networks (3 seeds), per-panel retraining (2 seeds), components
for s in 1 2 3; do PAUG=0.5 THIN=1 GDROP=0.5 DAUG=1 SEED=$s python scripts/train/train_gen.py pm_full_s$s 24000; done
for g in v1_377 breast_280 crlm_462 crlm_319 x5k; do for s in 1 2; do
  PANEL=$g PAUG=0.5 THIN=1 GDROP=0.5 DAUG=1 SEED=$s python scripts/train/train_gen.py pm_${g}_s$s 24000; done; done
python scripts/data/make_pca_emb.py
NODAPI=1 PAUG=0.5 THIN=1 GDROP=0.5 DAUG=1 SEED=1 python scripts/train/train_gen.py abl_nodapi_full 24000
NONICHE=1 PAUG=0.5 THIN=1 GDROP=0.5 DAUG=1 SEED=1 python scripts/train/train_gen.py abl_noniche_full 24000
PCAEMB=1 PAUG=0.5 THIN=1 GDROP=0.5 DAUG=1 SEED=1 python scripts/train/train_gen.py abl_pca_full 24000
ENS="tx_mix3=pm_full_s1+pm_full_s2+pm_full_s3;tx_c1r1=fin_clean_s1+fin_rand_s1" python scripts/benchmark/run_bench_gen.py \
  results/bench/ablation pm_full_s1,pm_full_s2,pm_full_s3,pm_s1=pm_{panel}_s1,pm_s2=pm_{panel}_s2,abl_nodapi_full,abl_noniche_full,abl_pca_full,fin_clean_s1,fin_rand_s1 data/cases/*
for r in pm_full_s1 abl_nodapi_full abl_noniche_full abl_pca_full; do python scripts/train/pred_kidney_full.py $r ${r}_img; done
python scripts/evaluation/image_metrics_runs.py results/bench/image_ablation.csv pm_full_s1_img,abl_nodapi_full_img,abl_noniche_full_img,abl_pca_full_img

# 8. image level (Fig. 3c,d; Supplementary S7, Table S6) and semi-synthetic benchmark (Fig. 4; Supplementary S1, Table S1)
python scripts/evaluation/image_metrics.py fin_ens6 refine_fin results/bench/image_metrics_kidney_fin.csv
# stage 2 vs stage 1 vs measured (kidney held-out stripes), with a stage 1 + matched-noise control: pixel W1, GLCM texture,
# per-cell features, patch SWD, Cellpose KID
python scripts/evaluation/stage2_realism.py results/stage2_realism
python scripts/evaluation/stage2_kid_cellpose.py results/stage2_realism/stage2_kid_cellpose.csv
python scripts/evaluation/stage2_realism_summary.py results/stage2_realism/stage2_realism_per_protein.csv \
  results/stage2_realism/stage2_kid_cellpose.csv results/stage2_realism/stage2_realism_summary.csv
python scripts/evaluation/image_metrics_ext.py results/bench/image_metrics_ext.csv $EXT
python scripts/data/sim_make.py rcc; python scripts/data/sim_make.py ccrcc
SIMROOT=data/sim SEED=1 python scripts/train/train_gen.py sim_plain 12000                                 # clean
SIMROOT=data/sim THIN=1 GDROP=0.5 DAUG=1 SEED=1 python scripts/train/train_gen.py sim_aug 12000           # randomized
SIMROOT=data/sim PAUG=0.5 THIN=1 GDROP=0.5 DAUG=1 SEED=1 python scripts/train/train_gen.py sim_mix 12000  # mixed
for PERT in none thin50 thin25 dilate2 merge20; do
  ENS="ens_cr=sim_plain+sim_aug" python scripts/evaluation/sim_eval.py results/sim/ens sim_plain,sim_aug,sim_mix $PERT
  python scripts/baselines/sim_sota.py prep $PERT
  python scripts/baselines/sim_sota.py run dgat $PERT; python scripts/baselines/sim_sota.py run spamosaic $PERT   # PyG env
  python scripts/baselines/sim_sota.py run totalvi $PERT                                                      # scvi env
  python scripts/baselines/nichetrans_official.py sim $PERT
  python scripts/baselines/sim_sota.py eval $PERT
done
python scripts/evaluation/sim_loc.py results/sim/loc    # share of each synthetic protein in its intended compartment
python scripts/figdata/sim_loc_crops.py                 # held-out window of Fig. 4a (all four compartments expressed)

# 9. case study: tumour-immune architecture of the five CRLM tumour sections (models trained on kidney; the same rule for
#    measured IMC, TxCellNet, NicheTrans, DGAT and the cognate mRNA)
python scripts/evaluation/crlm_case.py results/case_crlm/final       # summary.csv, profiles.csv, thresholds.csv, map_<section>.npz
```

`txcellnet/bench.py` holds the data access shared by all methods (cells, splits, targets, scoring), so that every method
is scored on exactly the same cells. Changes to the published methods needed for our data (for example DGAT graphs built in
spatial tiles without the 700-gene cell filter, SpaMosaic averaging standardized targets, totalVI pseudo-counts for imaging
intensities, and training-cell subsampling for SpaMosaic and totalVI) are listed in Supplementary Section S2 and in each
script's docstring. NicheTrans uses the official model class with the training of its Xenium tutorial
(`nichetrans_official.py`; each step is replayed as a CUDA graph of the same computation, checked against eager execution
by `nt_graph_equiv.py`), and each method keeps its own gene selection (DGAT: genes detected in >= 2.5% of training cells;
SpaMosaic: its 5,000 highly variable genes; totalVI: the tutorial's 4,000 seurat_v3 highly variable genes).
`grid_eval.py` reuses the DGAT functions of `dgat_bench.py` with pseudo-spots as nodes and checks that its cell-level scores
reproduce the main benchmark.

### Figures and tables
The figure scripts read one directory `FIGDATA`, collected as follows (the image windows go to `results/windows`):

```bash
F=FIGDATA
python scripts/tables/bench_tables.py within $F/tables; python scripts/tables/bench_tables.py cross $F/tables   # also Tables S3, S4
mkdir -p $F/final && cp -r results/bench/xsec/summary $F/final/summary
python scripts/figdata/crops_kidney.py rcc fin_ens6 refine_fin results/windows/crops_rcc.npz 240
python scripts/figdata/crlm_window.py results/windows crlm_tumor3_A3
python scripts/figdata/sim_region.py results/windows
python scripts/figdata/allmethods_extract.py $F results/windows; python scripts/figdata/allmethods_cognate.py $F
cp results/windows/crops_rcc.npz results/bench/image_metrics_kidney_fin.csv $F/
cp data/xenium_protein_rcc/pilot_L2/runs/refine_fin/metrics.csv $F/refine_metrics.csv
cp results/stage2_realism/stage2_realism_per_protein_all.csv results/stage2_realism/stage2_realism_summary.csv $F/
cp -r results/sim/ens $F/sim_ens; cp -r results/sim/sota $F/sim_sota; cp -r results/case_crlm/final $F/case_crlm
mkdir -p $F/sim_loc && cp results/sim/loc/sim_loc_rows.csv results/sim/loc/crops_common_rcc.npz $F/sim_loc/
python scripts/evaluation/summarize_final.py $F/ablation_all.csv txcellnet \
  pm_full_s1,pm_full_s2,pm_full_s3,pm_s1,pm_s2,$FIN,abl_nodapi_full,abl_noniche_full,abl_pca_full,tx_mix3,tx_c1r1
python scripts/figdata/protein_table.py $F nichetrans_off_e3
python scripts/figdata/per_protein_rows.py $F    # per-protein panel of Fig. 2b and Table S13
bash figures/make_all.sh $F figures_out    # SVG, PDF, PNG preview and the plotted numbers (CSV) of every figure
```

Supplementary tables:
```bash
python scripts/tables/sim_table.py results/tables/table_sim.tex
python scripts/tables/sim_loc_table.py results/sim/loc/sim_loc_rows.csv results/tables/table_sim_loc   # localization part of Table S1 (.tex, _zh.tex, _data.csv)
python scripts/evaluation/robustness.py results/bench/robustness
python scripts/tables/robust_tex.py results/bench/robustness/robustness.json results/tables/table_robust.tex
python scripts/tables/image_ext_table.py results/bench/image_metrics_ext.csv results/tables/table_image_ext.tex
python scripts/tables/case_table.py results/case_crlm/final/summary.csv results/bench/own/bench_cases.csv results/tables/table_case.tex
```

| Paper | Script(s) |
|---|---|
| Fig. 1 (method overview) | drawn by hand (`TxCellNet.png`) |
| Fig. 2 (three settings on identical test cells; per protein on kidney and CRLM; cross-dataset group means) | `scripts/tables/bench_tables.py`, `scripts/evaluation/xsec_summary.py`, `scripts/figdata/per_protein_rows.py`, `figures/perprotein.py`, `figures/fig2_bench.py` |
| Fig. 3 (images; within-cell patterns; stage-2 realism with matched-noise control) | `scripts/figdata/crops_kidney.py`, `allmethods_extract.py`, `scripts/evaluation/image_metrics.py`, `stage2_realism.py`, `stage2_kid_cellpose.py`, `stage2_realism_summary.py`, `figures/fig3_images.py` |
| Fig. 4 (semi-synthetic benchmark by compartment) | `scripts/evaluation/sim_eval.py`, `scripts/baselines/sim_sota.py`, `scripts/evaluation/sim_loc.py`, `scripts/figdata/sim_loc_crops.py`, `figures/fig4_sim.py` |
| Fig. 5 (case study: tumour-immune architecture of liver metastases) | `scripts/evaluation/crlm_case.py`, `figures/fig5_case.py` |
| Fig. S1 (ablations, stage 2) | `scripts/evaluation/summarize_final.py`, `scripts/train/train_refine.py`, `figures/figS1_ablation.py` |
| Stage-2 realism panel (stage 2 vs stage 1 vs measured, matched-noise control; `figS1b_stage2_realism`) | `scripts/evaluation/stage2_realism.py`, `stage2_kid_cellpose.py`, `stage2_realism_summary.py`, `figures/figS1b_stage2_realism.py` |
| Fig. S2 (which proteins transfer; kidney cross-patient column: `x_rcc_only`, `x_ccrcc_only`) | `scripts/figdata/protein_table.py`, `figures/figS2_proteins.py` |
| Fig. S3 (images of every method) | `scripts/figdata/crlm_window.py`, `allmethods_extract.py`, `allmethods_cognate.py`, `figures/figS3_allmethods.py` |
| Fig. S4 (three settings per group) | `scripts/evaluation/xsec_summary.py`, `figures/figS4_tiers.py` |
| Fig. S5 (group means of every method) | `scripts/tables/bench_tables.py`, `figures/figS5_tables.py` |
| Fig. S6 (paired comparison per section, three settings) | `scripts/tables/bench_tables.py`, `scripts/evaluation/xsec_summary.py`, `figures/figS6_paired.py` |
| Table 1 (benchmark sections) | `scripts/benchmark/own_only.py` (`bench_cases.csv`) |
| Table S1 (semi-synthetic benchmark; per-channel localization) | `scripts/tables/sim_table.py`, `scripts/evaluation/sim_loc.py`, `scripts/tables/sim_loc_table.py` |
| Table S2 (readouts, stain-cocktail weights) | `scripts/data/fit_cocktail.py` |
| Tables S3, S4 (per section: cross-dataset / same-section setting) | `scripts/tables/bench_tables.py cross` / `within` (`table_cross.tex`, `table_within.tex`) |
| Table S5 (Pearson, AUROC, single networks) | `scripts/evaluation/robustness.py`, `scripts/tables/robust_tex.py` |
| Table S6 (image level, external sections) | `scripts/evaluation/image_metrics_ext.py`, `scripts/tables/image_ext_table.py` |
| Table S7 (other-section setting) | `scripts/evaluation/xsec_summary.py`, `scripts/tables/xsec_table_tex.py` |
| Table S8 (spot-level evaluation) | `scripts/baselines/grid_eval.py`, `scripts/evaluation/grid_summary.py` |
| Table S9 (case study, per section) | `scripts/evaluation/crlm_case.py`, `scripts/tables/case_table.py` |
| Table S11 (data sources) | `data/README.md` |
| Table S13 (per protein and method on kidney and CRLM: the values of Fig. 2b) | `scripts/figdata/per_protein_rows.py`, `figures/perprotein.py`, `figures/table_per_protein.py` (run by `figures/make_all.sh`) |

## Pretrained models
All networks of the paper are trained with the commands above.

## Acknowledgements
We thank the developers of NicheTrans ([GitHub](https://github.com/YSTLab/NicheTrans)), DGAT
([GitHub](https://github.com/osmanbeyoglulab/DGAT)), SpaMosaic ([GitHub](https://github.com/JinmiaoChenLab/SpaMosaic)),
totalVI and scvi-tools ([GitHub](https://github.com/scverse/scvi-tools)), GenePT ([GitHub](https://github.com/yiqunchen/GenePT)),
Cellpose ([GitHub](https://github.com/MouseLand/cellpose)), readimc ([GitHub](https://github.com/BodenmillerGroup/readimc)),
Scanpy ([GitHub](https://github.com/scverse/scanpy)), PyTorch ([GitHub](https://github.com/pytorch/pytorch)) and PyTorch
Geometric ([GitHub](https://github.com/pyg-team/pytorch_geometric)), and the authors of the public datasets (10x Genomics;
Eapen et al.; Duchini et al.).

## License
This project is released under the [MIT License](LICENSE).

## Citation
Citation will be added.
