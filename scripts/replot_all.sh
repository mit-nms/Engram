#!/usr/bin/env bash
set -euo pipefail

DEFAULT_URL="https://zenodo.org/records/20015246/files/engram_paper_results.zip"

if [ $# -gt 1 ]; then
    echo "Usage: $0 [<input>]"
    echo "  With no argument, downloads from: $DEFAULT_URL"
    echo "  Otherwise <input> may be one of:"
    echo "    - a directory containing the unzipped results (e.g. .../engram_paper_results)"
    echo "    - a local .zip file (will be extracted next to it)"
    echo "    - an http(s) URL to a .zip (e.g. a Zenodo file URL)"
    exit 1
fi

input="${1:-$DEFAULT_URL}"

# Resolve input -> results_dir, downloading/extracting as needed.
case "$input" in
    http://*|https://*)
        dl_dir="$(realpath "$(dirname "$0")")/_downloaded_results"
        mkdir -p "$dl_dir"
        zip_path="$dl_dir/$(basename "${input%%\?*}")"
        extract_dir="$dl_dir/extracted"
        if [ -d "$extract_dir" ] && [ -n "$(ls -A "$extract_dir" 2>/dev/null)" ]; then
            echo "Using existing extraction: $extract_dir (skipping download and unzip)"
        else
            if [ -f "$zip_path" ] && unzip -tq "$zip_path" >/dev/null 2>&1; then
                echo "Using existing download: $zip_path"
            else
                echo "Downloading $input -> $zip_path (resumable)..."
                until curl -L --fail --progress-bar -C - -o "$zip_path" "$input"; do
                    echo "Download interrupted; retrying in 10s..."; sleep 10
                done
            fi
            mkdir -p "$extract_dir"
            echo "Unzipping into $extract_dir..."
            unzip -q "$zip_path" -d "$extract_dir"
        fi
        ;;
    *.zip)
        zip_path=$(realpath "$input")
        if [ ! -f "$zip_path" ]; then
            echo "Error: zip file not found: $zip_path"; exit 1
        fi
        extract_dir="$(dirname "$zip_path")/$(basename "$zip_path" .zip)_extracted"
        if [ -d "$extract_dir" ] && [ -n "$(ls -A "$extract_dir" 2>/dev/null)" ]; then
            echo "Using existing extraction: $extract_dir (skipping unzip)"
        else
            mkdir -p "$extract_dir"
            echo "Unzipping $zip_path -> $extract_dir..."
            unzip -q "$zip_path" -d "$extract_dir"
        fi
        ;;
    *)
        extract_dir=$(realpath "$input")
        if [ ! -d "$extract_dir" ]; then
            echo "Error: input is not a directory, .zip, or URL: $input"; exit 1
        fi
        ;;
esac

# After extraction the results may be at $extract_dir itself or one level down.
if [ -d "$extract_dir/paper_json" ]; then
    results_dir="$extract_dir"
else
    nested=$(find "$extract_dir" -maxdepth 2 -type d -name paper_json -print -quit)
    if [ -n "$nested" ]; then
        results_dir=$(dirname "$nested")
    else
        echo "Error: could not locate paper_json/ under $extract_dir"; exit 1
    fi
fi
results_dir=$(realpath "$results_dir")
echo "Using results_dir=$results_dir"

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
