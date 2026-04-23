current_dir=$(pwd)
task_prompt_path=$current_dir/SystemBench/ADRS/cloudcast/deepagents_files/task_prompt.txt
evaluator_path=$current_dir/SystemBench/ADRS/cloudcast
base_save_dir=${GLIA_RESULTS_BASE_DIR:-$current_dir/results}

model="o3"

# Run evolution optimization
for i in {1..10}; do
    python -m Architect.main --method evolution --model $model \
        --task_prompt_path $task_prompt_path \
        --evaluator_path $evaluator_path \
        --num_generations 100 \
        --results_dir $base_save_dir/evolution_results/cloudcast_evolution_results_${model}_run${i}
done
