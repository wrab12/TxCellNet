#!/usr/bin/env bash
# Draw every data figure of the paper (Figs 2-5, Supplementary Figs S1-S5) from the collected figure data.
# usage: bash figures/make_all.sh FIGDATA OUT
#   FIGDATA: directory filled by the "Figures and tables" commands of README.md; OUT: output directory
# Each figure is written as OUT/<name>.svg, .pdf (TrueType fonts), .png (preview) and OUT/<name>_data_*.csv (plotted numbers).
set -euo pipefail
if [ "$#" -ne 2 ]; then echo "usage: bash figures/make_all.sh FIGDATA OUT" >&2; exit 1; fi
D=$(cd "$1" && pwd)
mkdir -p "$2"; O=$(cd "$2" && pwd)
H=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python}
"$PY" "$H/fig2_bench.py" "$D" "$O/fig2_bench"
"$PY" "$H/fig3_images.py" "$D" "$O/fig3_images"
"$PY" "$H/fig4_sim.py" "$D" "$O/fig4_sim"
"$PY" "$H/fig5_case.py" "$D/case_crlm" "$O/fig5_case"
"$PY" "$H/figS1_ablation.py" "$D/ablation_all_per_dataset.csv" "$D/refine_metrics.csv" "$D/crops_rcc.npz" "$O/figS1_ablation"
"$PY" "$H/figS2_proteins.py" "$D/protein_table.csv" "$O/figS2_proteins"
"$PY" "$H/figS3_allmethods.py" "$D" "$O/figS3_allmethods"
"$PY" "$H/figS4_tiers.py" "$D/final/summary/xsec_groups.csv" "$O/figS4_tiers"
"$PY" "$H/figS5_tables.py" "$D" "$O/figS5_tables"
