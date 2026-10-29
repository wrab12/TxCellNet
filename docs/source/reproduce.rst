Reproducing the paper
=====================

The `Reproducing the paper <https://github.com/wrab12/TxCellNet#reproducing-the-paper>`_ section of the README lists
every command of the paper in order. All scripts run from the repository root, read ``data/`` (see :doc:`data`) and write
``results/``; each script documents its inputs, outputs and options in its docstring. Commands marked ``# PyG env`` or
``# scvi env`` run in the environments of the compared methods (see :doc:`installation`). No pretrained weights are
distributed: every network of the paper is trained by these commands.

The paper evaluates three settings, which are also the script modes: same-section (``within``), other-section
(``xsec``) and cross-dataset (``cross``, trained on the two kidney sections only). Every method is scored on the same
cells through ``txcellnet/bench.py``.

Steps
-----

1. **Kidney training grids** (``scripts/data/prep_kidney_grid.py``, ``prep_kidney_tx.py``, ``make_kidney_splits.py``):
   DAPI, the 27 protein images, cells and transcripts of both kidney sections on a 0.85 µm/px grid, the split into eight
   vertical stripes (odd stripes held out) and the vendor per-cell protein values of the held-out cells.
2. **Benchmark sections** (``scripts/data/prep_case_*.py`` and the other scripts in ``scripts/data``): one directory per
   section under ``data/cases``, panel groups, gene lists, stain-cocktail weights and image targets.
3. **Cross-dataset setting**: three clean and three randomized stage-1 networks on the kidney training stripes
   (``scripts/train/train_gen.py``, 24,000 iterations each), stage 2 on the six-network ensemble
   (``scripts/train/pred_kidney_full.py``, ``scripts/train/train_refine.py``), TxCellNet on every section
   (``scripts/benchmark/run_bench_gen.py``) and the compared methods (``scripts/baselines``). The kidney held-out stripes
   scored here are also the kidney rows of the same-section setting.
4. **Same-section setting**: one network per external section (``scripts/train/train_case.py``) and the compared
   methods trained on the same section.
5. **Other-section setting**: 50 section pairs (``scripts/benchmark/xsec_txcellnet.py`` and the baseline scripts), with
   single-section networks for the kidney pair, then ``scripts/evaluation/xsec_summary.py``, which compares the three
   settings on identical test cells.
6. **Spot-level evaluation** (Supplementary Section S9): ``scripts/baselines/grid_eval.py``,
   ``scripts/evaluation/grid_summary.py``.
7. **Ablations** (Supplementary Section S3): mixed reference networks and networks without DAPI, without the
   cell-neighborhood transformer or with fixed PCA gene vectors.
8. **Image level and semi-synthetic benchmark**: image metrics, stage-2 realism against measured images with a
   matched-noise control (``scripts/evaluation/stage2_realism.py``), and the semi-synthetic benchmark of Fig. 4
   (``scripts/data/sim_make.py``, ``scripts/evaluation/sim_eval.py``).
9. **Case study**: tumor-immune architecture of the five colorectal liver metastasis sections
   (``scripts/evaluation/crlm_case.py``).

Figures and tables
------------------

The figure scripts read one directory of collected results; ``figures/make_all.sh`` draws every figure (SVG, PDF, PNG
preview and the plotted numbers as CSV). The README lists how to collect that directory, the commands of the
supplementary tables, and the scripts behind each figure and table.
