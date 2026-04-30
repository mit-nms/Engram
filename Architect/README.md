# Architect

The optimization framework that powers Engram and all baseline methods. It provides the CLI, task loading, LLM interface, and base classes for optimization methods.

See the [examples README](../examples/README.md) for how to run each method.

---

## Directory Structure

```
Architect/
├── main.py          # CLI entry point
├── task.py          # Task class: wraps a task prompt + evaluator
├── types.py         # Shared type definitions
├── utils.py         # Logging and directory utilities
├── methods/         # All optimization method implementations
│   ├── common.py    # OptimizationMethod base class
│   ├── agentic_handoff.py   # Engram
│   ├── evolution.py         # Evolution of Heuristics
│   ├── single_agent.py      # Single-agent baseline (Glia)
│   ├── ...                  # Other methods
│   ├── deepagents_utils/    # System prompts and agent configs
│   └── handoff_utils/       # Archive and Research Digest logic
├── llm/             # LLM interface (OpenAI API) and prompt compilation
└── openevolve/      # Adopted from github.com/algorithmicsuperintelligence/openevolve
```

---

## Adding a New Method

1. Create a new file in `methods/` (e.g., `methods/my_method.py`).

2. Subclass `OptimizationMethod` from `methods/common.py` and implement the `optimize()` method:

```python
from Architect.methods.common import OptimizationMethod

class MyMethod(OptimizationMethod):
    def __init__(self, task, model, results_dir, debug=True, **kwargs):
        super().__init__(task, model, results_dir, debug)
        # your init here

    def optimize(self):
        # Use self.task, self.architect, self.log_dir, etc.
        # Return a results dict with at minimum {"best_solution": {"score": ..., "code": ...}}
        pass
```

3. Register the method in `main.py` by adding it to the method dispatch logic.

4. Create an example script in `examples/` following the pattern of existing scripts (see `examples/handoff_example_usage.py`).
