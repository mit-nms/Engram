#!/bin/bash
# Regenerate all plots that live under Camera_ready_Glia/paper_plots_glia/
# Each invocation uses plot_methods_paper.py with an --output-dir whose
# basename is reused as the prefix of every generated plot file.

set -euo pipefail

scripts_dir=$(realpath "$(dirname "$0")")
out_root=$scripts_dir/Camera_ready_Glia/paper_plots_glia
plot=$scripts_dir/plot_methods_paper.py

mkdir -p "$out_root"

# Single-context glia
python "$plot" \
    -a "$scripts_dir/all_methods_paper.json" \
    -o "$out_root/paper_plots_15" \
    --max-num-sims 15 \
    --problem-name vidur \
    --skip-multi \
    --cut-length 15

# More budget with MCG methods
python "$plot" \
    -a "/data1/pantea/Glia/scripts/all_methods_paper.json" \
    -o "$out_root/paper_plots_3" \
    --max-num-sims 100 \
    --problem-name vidur \
    --cut-length 15

# Vidur ablation: Evolution vs. Seed+Evolution
python "$plot" \
    -a "/data1/pantea/Glia/scripts/paper_json/vidur_ablation.json" \
    -o "$out_root/vidur_ablation" \
    --max-num-sims 100 \
    --problem-name vidur \
    --skip-multi

# One-shot LLM-alone comparison across models
python "$plot" \
    -a "/data1/pantea/Glia/scripts/llm_alone.json" \
    -o "$out_root/llm_alone" \
    --max-num-sims 100 \
    --problem-name vidur \
    --skip-multi
