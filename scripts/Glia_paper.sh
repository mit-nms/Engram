#!/bin/bash

root_dir=$(realpath "$(dirname "$0")/..")
problem_name="vidur"
num_runs=10
model="o3"
extended=true


task_prompt_path=$root_dir/SystemBench/vidur/deepagents_files/task_prompt.txt
evaluator_path=$root_dir/SystemBench/vidur
initial_program_path=$root_dir/SystemBench/vidur/deepagents_files/llq_scheduler.py
openevolve_config_path=$root_dir/SystemBench/vidur/openevolve_files/config.yaml


common_args="--task_prompt_path $task_prompt_path --evaluator_path $evaluator_path"
common_args="$common_args --debug"
common_args="$common_args --num_generations 100"
paper_results_dir=$root_dir/results_${problem_name}

export GLIA_RESULTS_BASE_DIR=$paper_results_dir

## Run OpenEvolve
python examples/openevolve_example_usage.py --problem_name $problem_name --model $model --num_runs $num_runs --config_path $openevolve_config_path

if [ "$extended" = true ]; then
    ## Run Glia
    python examples/single_agent_example_usage.py --problem_name $problem_name --model $model --num_runs $num_runs --enable_continue_message --agent_timeout 60

    ## Run FunSearch
    for i in $(seq 1 $num_runs); do
        python -m Architect.main $common_args --method best-shot --results_dir ${paper_results_dir}/FunSearch_${model}_${i} --model $model &
    done
    wait

    ## Run Evolution of Heuristics
    for i in $(seq 1 $num_runs); do
        python -m Architect.main $common_args --method evolution --results_dir ${paper_results_dir}/EOH_${model}_${i} --model $model &
    done
    wait
fi

## Aggregate all results into a single all_methods JSON, then plot
paper_json_dir=$root_dir/paper_json_${problem_name}
paper_plots=$root_dir/paper_plots_${problem_name}
auto_json=$paper_json_dir/${problem_name}_auto_${model}.json
mkdir -p "$paper_json_dir" "$paper_plots"

# Detect Handoff and Single Agent (OpenEvolve handled below — its layout has an extra cloudcast/openevolve_output/ nesting that build_all_methods_json.py doesn't traverse)
python $root_dir/scripts/build_all_methods_json.py \
    --output "$auto_json" \
    "$paper_results_dir"

# Append FunSearch (best-shot) and EOH (evolution) entries — not handled by build script
python - <<PYEOF
import json, re, subprocess, sys
from pathlib import Path

auto_json = Path("$auto_json")
results_dir = Path("$paper_results_dir")
scripts_dir = Path("$root_dir/scripts")
model = "$model"
problem = "$problem_name"
extended = "$extended" == "true"

with auto_json.open() as f:
    data = json.load(f)

# Rename default labels from build_all_methods_json.py to paper-style labels
rename = {f"Glia Handoff ({model})": f"Handoff ({model})"}
if extended:
    rename[f"Glia Single Agent ({model})"] = f"Glia ({model})"
for old, new in rename.items():
    if old in data:
        data[new] = data.pop(old)

def collect_last_gen(prefix, label):
    paths = []
    for run_dir in sorted(results_dir.glob(f"{prefix}_{model}_*")):
        logs = run_dir / problem / "logs"
        if not logs.exists():
            continue
        cands = []
        for p in logs.iterdir():
            if p.suffix != ".json" or "usage_stats" in p.name:
                continue
            m = re.search(r"_(\d+)gen\.json$", p.name)
            if m:
                cands.append((int(m.group(1)), p))
        if cands:
            cands.sort(reverse=True)
            paths.append(str(cands[0][1].resolve()))
    if paths:
        data[label] = sorted(paths)

if extended:
    collect_last_gen("FunSearch", f"Best Shot ({model})")
    collect_last_gen("EOH", f"Evolution ({model})")

# OpenEvolve: layout is results/{problem}_openevolve_results/{problem}_openevolve_results_{i}/{problem}/openevolve_output/generated_programs
oe_paths = []
oe_base = results_dir / f"{problem}_openevolve_results"
for run_dir in sorted(oe_base.glob(f"{problem}_openevolve_results_*")):
    oe_dir = run_dir / problem / "openevolve_output"
    if not (oe_dir / "generated_programs").exists():
        continue
    agg = oe_dir / "aggregated_results.json"
    if not agg.exists():
        r = subprocess.run([sys.executable, str(scripts_dir / "aggregate_openevolve_logs.py"),
                            str(oe_dir), str(agg)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"OpenEvolve aggregation failed for {run_dir.name}: {r.stderr}", file=sys.stderr)
            continue
    oe_paths.append(str(agg.resolve()))
if oe_paths:
    data[f"OpenEvolve ({model})"] = sorted(oe_paths)

with auto_json.open("w") as f:
    json.dump(data, f, indent=2)
print(f"Wrote {auto_json}")
PYEOF

# Generate plots
python $root_dir/scripts/plot_methods_paper.py \
    -a "$auto_json" \
    -o "$paper_plots/${problem_name}_auto_${model}" \
    --max-num-sims 100 \
    --problem-name $problem_name
