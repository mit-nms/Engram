# SystemBench

Benchmark problems and evaluators for Engram and all baseline methods. Each problem defines a simulation environment and an evaluator that scores candidate algorithms.

---

## Available Benchmarks

### Engram Paper Problems

| Problem | Directory | Domain |
|:--------|:----------|:-------|
| Multi-Cloud Multicast | [`ADRS/cloudcast/`](ADRS/cloudcast/) | Networking |
| LLM Request Routing | [`vidur/`](vidur/) | ML Systems |
| KV Cache Reuse | [`ADRS/llm_sql/`](ADRS/llm_sql/) | Databases |

### Additional Benchmarks

- **[ADRS](ADRS/README.md)** -- Scheduling, load balancing, telemetry repair, and more. Adopted from the [ADRS project](https://github.com/UCB-ADRS/ADRS).
- **[FrontierCS](FrontierCS/README.md)** -- 100+ problem instances across systems, ML, security, and scientific computing. From the [Frontier-CS project](https://github.com/FrontierCS/Frontier-CS).
- **[vidur](vidur/)** -- LLM inference simulator. Submodule from [mit-nms/vidur_simulator](https://github.com/mit-nms/vidur_simulator).

---

## Adding a New Problem

### 1. Create an Evaluator

Create a directory under `SystemBench/` for your problem and implement an evaluator class inheriting from [`evaluator.py`](evaluator.py). The only required method is `run_simulation()`:

```python
from SystemBench.evaluator import Evaluator

class MyEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        # Register the function/class the agent will optimize
        # e.g., self._functions["my_func"] = Function(...)

    def run_simulation(self, design_config, scenario):
        # Run your simulation, return dict with:
        #   success (bool), score (float), metrics (dict), error (str)
        pass
```

Optionally, you can also implement:
- `get_baselines()` -- return `[(description, code), ...]` for baseline comparisons at startup
- `analyze_results(results)` -- custom logging/analysis after simulation
- `get_system_model()` -- return a system description string (legacy prompt compilation)

### 2. Write a Task Prompt

Create `deepagents_files/task_prompt.txt` in your problem directory describing:
- What the system does
- What the agent needs to optimize
- Input/output format of the target function
- Constraints and domain knowledge

### 3. (Optional/Recommended) Provide an Initial Program

Add an `initial_program.py` with a baseline or sample implementation. When passed via `--initial_program_path`, the agent sees this as a starting point instead of writing code from scratch.

### 4. Run

You can run directly via the CLI without creating an example script:

```bash
python -m Architect.main \
    --method agentic_handoff \
    --model o3 \
    --task_prompt_path SystemBench/my_problem/deepagents_files/task_prompt.txt \
    --evaluator_path SystemBench/my_problem \
    --results_dir results/my_problem_run1 \
    --debug
```

Optionally, add your problem to `examples/_common.py` in `resolve_problem_config()` to use it with the example scripts. See the [examples README](../examples/README.md).
