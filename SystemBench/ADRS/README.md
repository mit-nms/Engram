# ADRS Benchmarks

Simulation environments adopted from the [ADRS project](https://github.com/UCB-ADRS/ADRS). Each subdirectory contains an evaluator that wraps an ADRS environment for use with the optimization framework.

### All environments

| Directory | Description |
|:----------|:------------|
| `cloudcast` | Cloud multicast data transfer |
| `eplb` | Expert-parallel load balancing |
| `cant-be-late` | Job scheduling |
| `cant-be-late-multi` | Multi-resource job scheduling |
| `txn_scheduling` | Transaction scheduling |
| `prism` | Resource management |
| `telemetry_repair` | Network telemetry repair |
| `llm_sql` | LLM-based SQL KV cache reuse |
| `llm_sql_col_merge` | LLM SQL column merge variant |
| `llm_sql_ggr_adrs` | LLM SQL GGR-ADRS variant |
| `sparse_attention` | Sparse attention optimization |
| `hp_quantization` | Hyperparameter quantization |

---

## Running an ADRS problem

```bash
python -m Architect.main \
    --method agentic_handoff \
    --model o3 \
    --task_prompt_path SystemBench/ADRS/cloudcast/deepagents_files/task_prompt.txt \
    --evaluator_path SystemBench/ADRS/cloudcast \
    --results_dir results/cloudcast_run1
```

Or use the example scripts with `--problem_name` (e.g., `--problem_name cloudcast`). See the [examples README](../../examples/README.md).

---

## Adding a new ADRS environment

1. Create a directory `SystemBench/ADRS/<your_env>/`
2. Add an `evaluator.py` with:
   - `TARGET_NAME` -- name of the function/class to optimize
   - `evaluate(program_path: str) -> dict` -- returns at least `{"combined_score": float}`
3. Add an `initial_program.py` with the baseline implementation
4. Add a task prompt at `deepagents_files/task_prompt.txt`
