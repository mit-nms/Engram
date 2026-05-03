#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <path-to-engram_paper_results>"
    echo "  e.g. $0 /data2/projects/pantea-work/Glia/engram_paper_results"
    exit 1
fi

results_dir=$(realpath "$1")
if [ ! -d "$results_dir" ]; then
    echo "Error: results directory does not exist: $results_dir"
    exit 1
fi

current_dir=$(realpath "$(dirname "$0")")
paper_plots="$current_dir/Camera_ready_Engram"
src_json_dir="$results_dir/paper_json"
json_dir="$current_dir/paper_json_rewritten"
old_prefix="/data2/projects/pantea-work/Glia/results"

if [ ! -d "$src_json_dir" ]; then
    fallback="$results_dir/paper_json"
    if [ -d "$fallback" ]; then
        echo "Note: $src_json_dir not found, using $fallback"
        src_json_dir="$fallback"
    else
        echo "Error: source paper_json directory not found."
        echo "  Looked in: $src_json_dir"
        echo "  And: $fallback"
        exit 1
    fi
fi

rm -rf "$json_dir"
mkdir -p "$json_dir"
mkdir -p "$paper_plots"

echo "Rewriting paper_json files: $src_json_dir -> $json_dir"
echo "  replacing prefix '$old_prefix' with '$results_dir'"

python3 - "$src_json_dir" "$json_dir" "$old_prefix" "$results_dir" <<'PY'
import json, os, sys

src_dir, dst_dir, old_prefix, new_prefix = sys.argv[1:5]
old_prefix = old_prefix.rstrip("/")
new_prefix = new_prefix.rstrip("/")

def rewrite(obj):
    if isinstance(obj, str):
        if obj.startswith(old_prefix + "/"):
            return new_prefix + obj[len(old_prefix):]
        return obj
    if isinstance(obj, dict):
        return {k: rewrite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite(v) for v in obj]
    return obj

skipped = []
for fname in sorted(os.listdir(src_dir)):
    if not fname.endswith(".json"):
        continue
    src = os.path.join(src_dir, fname)
    dst = os.path.join(dst_dir, fname)
    try:
        with open(src) as f:
            data = json.load(f)
    except Exception as e:
        skipped.append(f"{fname}: {e}")
        continue
    with open(dst, "w") as f:
        json.dump(rewrite(data), f, indent=2)

if skipped:
    print("Skipped (parse errors):")
    for s in skipped:
        print(f"  - {s}")
PY

echo "Done rewriting. Plotting..."
echo

# Cloudcast
python $current_dir/plot_methods_paper.py -a $json_dir/cloudcast_my_prompt.json -o $paper_plots/cloudcast_my_prompt --max-num-sims 100 --problem-name cloudcast
python $current_dir/plot_methods_paper.py -a $json_dir/cloudcast_simp_prompt.json -o $paper_plots/cloudcast_simp_prompt --max-num-sims 100 --problem-name cloudcast
python $current_dir/plot_methods_paper.py -a $json_dir/cloudcast_simp_prompt_5.2.json -o $paper_plots/cloudcast_simp_prompt_5.2 --max-num-sims 100 --problem-name cloudcast
python $current_dir/plot_methods_paper.py -a $json_dir/cloudcast_baseline_my_o3.json -o $paper_plots/cloudcast_baseline_my_o3 --max-num-sims 100 --problem-name cloudcast
python $current_dir/plot_methods_paper.py -a $json_dir/cloudcast_motivation.json -o $paper_plots/cloudcast_motivation --max-num-sims 100 --problem-name cloudcast
python $current_dir/plot_methods_paper.py -a $json_dir/cloudcast_ablation.json -o $paper_plots/cloudcast_ablation --max-num-sims 100 --problem-name cloudcast
python $current_dir/plot_methods_paper.py -a $json_dir/cloudcast_simp_o3.json -o $paper_plots/cloudcast_simp_o3 --max-num-sims 100 --problem-name cloudcast
python $current_dir/plot_methods_paper.py -a $json_dir/cloudcast_simp_gpt5.2.json -o $paper_plots/cloudcast_simp_gpt5.2 --max-num-sims 100 --problem-name cloudcast
python $current_dir/plot_methods_paper.py -a $json_dir/cloudcast_system_prompt_ablation.json -o $paper_plots/cloudcast_system_prompt_ablation --max-num-sims 100 --problem-name cloudcast
python $current_dir/plot_methods_paper.py -a $json_dir/cloudcast_simp_gpt5.2_all.json -o $paper_plots/cloudcast_simp_gpt5.2_all --max-num-sims 100 --problem-name cloudcast


# Vidur
python $current_dir/plot_methods_paper.py -a $json_dir/vidur.json -o $paper_plots/vidur_baseline --max-num-sims 100 --problem-name vidur
