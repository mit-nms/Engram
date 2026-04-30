# Examples

Ready-to-run scripts for every optimization method in the repository. Each script supports multiple benchmark problems via `--problem_name`.

---

## Engram (Agentic Handoff)

**[`handoff_example_usage.py`](handoff_example_usage.py)** -- The main method from the [Engram paper](https://arxiv.org/abs/2603.21321). Sequential agents hand off knowledge through a persistent Archive and Research Digest.

```bash
python examples/handoff_example_usage.py \
    --problem_name cloudcast \
    --model o3 \
    --max_agents 5 \
    --agent_timeout 30
```

| Flag | Description |
|:-----|:------------|
| `--problem_name` | Benchmark problem (e.g., `vidur`, `cloudcast`, `llm_sql`, `eplb`, `fcs_alg_<id>`, `fcs_res_<id>`) |
| `--model` | LLM model to use |
| `--max_agents` | Number of sequential agents |
| `--agent_timeout` | Wall-clock timeout per agent (minutes) |

Run `python examples/handoff_example_usage.py --help` for all options.

---

## Available Problems

All example scripts accept a `--problem_name` argument. The three core Engram paper problems:

| Problem name | Paper name | Domain |
|:-------------|:-----------|:-------|
| `cloudcast` | Multi-Cloud Multicast | Networking |
| `vidur` | LLM Request Routing | ML Systems |
| `llm_sql` | KV Cache Reuse | Databases |

Additional ADRS problems: `eplb`, `cant-be-late`, `cant-be-late-multi`, `txn_scheduling`, `prism`, `telemetry_repair`, `llm_sql_ggr_ours`, `llm_sql_ggr_adrs`, `llm_sql_col_merge`

FrontierCS problems: `fcs_alg_<id>` (algorithmic track) and `fcs_res_<id>` (research track)

---

## Other Methods

Baseline and alternative optimization methods for comparison with Engram:

- **[`example_usage.py`](example_usage.py)** -- Evolution of Heuristics (EoH). Classic evolutionary optimization with seed population.
- **[`openevolve_example_usage.py`](openevolve_example_usage.py)** -- OpenEvolve. LLM-based evolutionary code optimization.
- **[`single_agent_example_usage.py`](single_agent_example_usage.py)** -- Single agent baseline. A non-supervisor implementation of [Glia](https://arxiv.org/abs/2510.27176); one long-running agent with no handoff.
- **[`tree_deepagents_example_usage.py`](tree_deepagents_example_usage.py)** -- Tree search over multiple DeepAgents runs, keeping best.
- **[`linear_agent_example_usage.py`](linear_agent_example_usage.py)** -- Sequential agents with code-only handoff (no Archive or Digest).
- **[`handoff_no_journal_example_usage.py`](handoff_no_journal_example_usage.py)** -- Engram ablation: Archive only, no Research Digest.