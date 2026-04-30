# Artifact Appendix

This document is the reviewer guide for the artifact accompanying the paper
*"Improving Coherence and Persistence in Agentic AI for System Optimization"*.

## Abstract

Designing high-performance system heuristics is a creative, iterative process requiring experts to form hypotheses and execute multi-step conceptual shifts. While Large Language Models (LLMs) show promise in automating this loop, they struggle with complex system problems due to two critical failure modes: evolutionary neighborhood bias and the coherence ceiling. Evolutionary methods often remain trapped in local optima by relying on scalar benchmark scores, failing when coordinated multi-step changes are required. Conversely, existing agentic frameworks suffer from context degradation over long horizons or fail to accumulate knowledge across independent runs.

We present Engram, an agentic researcher architecture that addresses these limitations by decoupling long-horizon exploration from the constraints of a single context window. Engram organizes exploration into a sequence of agents that iteratively design, test, and analyze mechanisms. At the conclusion of each run, an agent stores code snapshots, logs, and results in a persistent Archive and distills high-level modeling insights into a compact, persistent Research Digest. Subsequent agents then begin with a fresh context window, reading the Research Digest to build on prior discoveries.

We find that Engram exhibits superior performance across diverse domains including multi-cloud multicast, LLM inference request routing, and optimizing KV cache reuse in databases with natural language queries

## Artifact check-list (meta-information)

- **Algorithm:** Engram (agentic handoff with persistent Archive and
  Research Digest), plus baselines: single-agent,
  FunSearch, Evolution of Heuristics, OpenEvolve.
- **Program:** Python 3.11 framework; candidate programs are generated
  by the agent at runtime and evaluated in-process.
- **Model:** OpenAI `o3` and `gpt-5.2` (the two models reported in the
  paper). Other OpenAI models are supported for
  exploration.
- **Data set:** Multi-Cloud Multicast throughput/profile tables
  (shipped in `SystemBench/ADRS/cloudcast/profiles/`); LLM Request
  Routing traces (generated via the Vidur simulator); KV Cache Reuse
  datasets (BIRD, PDMX, beer, movies, products — shipped in
  `SystemBench/ADRS/llm_sql/datasets/`). See §Data sets below.
- **Run-time environment:** Linux x86_64, Python 3.11, Docker Engine
  reachable without sudo, OpenAI API access.
- **Hardware:** Any modern x86_64 machine. No GPU required for the
  three core benchmarks.
- **Output:** Per-run JSON logs under `results/` containing the best
  solution, score progression, full reasoning traces, and cost
  accounting.
- **Experiments:** Agentic (Engram, single-agent, tree), evolutionary
  (FunSearch, EoH, OpenEvolve), one-shot.
- **Approximate time needed to prepare workflow:** ~15 minutes
  (install + smoke test).
- **Approximate time needed to complete experiments:** minutes
  (smoke test) to hours (paper-scale sweep). See
  §Experiment workflow.
- **Publicly available?** Yes — TODO (link)

## Description

### How to access

Clone the public repository and its submodules:

```bash
git clone --recurse-submodules https://github.com/mit-nms/Engram.git
cd Engram
git checkout main   # or the release branch tip
```

The repository is MIT-licensed (`LICENSE`).

### Hardware dependencies

- CPU: any x86_64 — no special instruction sets required.
- RAM: 8 GB recommended.
- Disk: Budget 10 GB total.
- Network: outbound HTTPS to `api.openai.com` and `docker.io`.

### Software dependencies

- Linux (tested on Debian 12 / Ubuntu 22.04).
- Python 3.11 (tested with 3.11.15).
- Docker Engine; the current shell user must be able to run
  `docker ps` without `sudo`. Engram's agents sandbox their shell
  tool inside a `python:3.12-alpine3.19` container.
- An OpenAI API key with access to `o3` (and `gpt-5.2` if reproducing
  the full paper sweep).
- All Python package dependencies are pinned in `requirements.txt`,
  `SystemBench/vidur/requirement.txt`, and the `openevolve/` editable
  install.

### Models

Engram and all baselines call OpenAI's chat-completions/responses API.
The paper reports results on `o3` and `gpt-5.2`. The artifact also
supports other OpenAI models for cheaper
exploration; see `Architect/pricing_table.py`.

## Installation

```bash
# Clone
git clone --recurse-submodules https://github.com/mit-nms/Engram.git
cd Engram

# Python env
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r SystemBench/vidur/requirement.txt
pip install -e Architect/openevolve/

# API key
export OPENAI_API_KEY="sk-..."

# Sanity-check Docker
docker ps
```

## Experiment workflow

The artifact is organized around one driver script per method, in
`examples/`. Every driver accepts `--problem_name` (one of `cloudcast`,
`vidur`, `llm_sql`, plus additional benchmarks listed in
`examples/README.md`) and `--model` (e.g. `o3`, `gpt-5.2`). Runs
produce JSON logs under `results/`.

### Smoke test (≤ 3 minutes, trivial cost)

```bash
bash scripts/smoke_test.sh
```

This runs one Engram agent for 2 minutes on `cloudcast` with `o3`,
preflight-checks Docker and the API key, and produces a single-iteration
result JSON. Successful completion prints `Smoke test finished` and a
non-zero best score; this confirms the install and the agent↔API↔Docker
plumbing.

### Single-benchmark Engram run (~1 hour per run)

```bash
python examples/handoff_example_usage.py \
    --problem_name cloudcast \
    --model o3 \
    --num_runs 1 \
    --max_agents 5 \
    --agent_timeout 30
```

Each of the three core benchmarks works the same way; swap
`--problem_name` for `vidur` or `llm_sql`.

### Baselines

```bash
# Single-agent (the "Glia" ablation)
python examples/single_agent_example_usage.py  --problem_name cloudcast --model o3 --num_runs 1 --agent_timeout 60

# Evolution of Heuristics
python -m Architect.main --method evolution --model o3 \
    --task_prompt_path SystemBench/ADRS/cloudcast/deepagents_files/task_prompt_direction.txt \
    --evaluator_path SystemBench/ADRS/cloudcast \
    --num_generations 100 \
    --results_dir results/eoh_cloudcast

# OpenEvolve
python examples/openevolve_example_usage.py --problem_name cloudcast --model o3 --num_runs 1
```

See `examples/README.md` for the other baselines (FunSearch, tree
search, linear agent, no-journal handoff).

### Paper-scale sweep (`scripts/paper.sh`)

`scripts/paper.sh` orchestrates the full "all methods × 3 runs × 100
generations" sweep for the problem selected inside the script (default
`llm_sql`). Editing `problem_name=...` at the top selects a
different benchmark.

By default the script runs **Engram** and **OpenEvolve** only. Pass
`--extended` to also run the Glia single-agent baseline, FunSearch, and
Evolution of Heuristics:

```bash
# Default (Engram + OpenEvolve)
bash scripts/paper.sh

# Full paper sweep (adds Glia, FunSearch, Evolution of Heuristics)
bash scripts/paper.sh --extended
```

Aggregation and plotting are performed automatically after all runs
finish, producing a combined JSON and figures under
`paper_json_<problem>/` and `paper_plots_<problem>/`.

### How a reviewer should verify the artifact works

1. `bash scripts/smoke_test.sh` exits 0 and prints a non-negative
   `Best score`. This alone demonstrates the software stack functions
   end-to-end.
2. Run one of the single-benchmark commands above for `cloudcast`,
   `vidur`, or `llm_sql`. In a single ~1-hour run, Engram typically:
   - improves over the baseline score within the first 2–3 agents;
   - reaches a score in the same order of magnitude as the "paper
     target" value in the table above (a single run is noisier than
     the 10-run median reported in the paper, so exact matching is not
     expected).
3. The run's results folder `results/...` contains the full
   iteration history and usage statistics. Reviewers can open it
   directly or browse runs visually with `scripts/experiment_viewer/`
   (Flask UI, `pip install flask plotly` first).

### How this compares to the paper

A single-seed AE run is expected
to land anywhere within the spread illustrated in the paper's box
plots. What is robust across single runs is the *ordering* of methods
— Engram consistently outperforms the one-shot, single-agent, and
evolutionary baselines in the paper, and that qualitative ordering
typically survives single-seed experiments too.

## Experiment customization

Every `examples/*_example_usage.py` driver exposes the same knobs
(see `--help`):

| Flag | Effect |
|:--|:--|
| `--num_runs N` | Number of independent seeds (default 10). |
| `--max_agents N` | Upper bound on the agent chain length (default 30). |
| `--agent_timeout MINUTES` | Per-agent wall-clock budget (default 60). |
| `--model NAME` | OpenAI model (default `o3`). `gpt-5.2`, `o4-mini`, etc., supported. |
| `--debug` | Stream agent turns to stdout; useful for reviewers wanting to see the agent "think". |

Global environment variables:

- `OPENAI_API_KEY` — required.
- `GLIA_RESULTS_BASE_DIR` — optional. If set, results will be saved in this directory.

Adding new benchmarks: see `SystemBench/README.md` for the evaluator
interface and `examples/README.md` for the example-script pattern.

## Methodology

Submission, reviewing, and badging follow the ACM Artifact Review and
Badging policy (v1.1, August 2020, <https://www.acm.org/publications/policies/artifact-review-and-badging-current>)
and the common artifact-evaluation guidelines hosted at
<https://sysartifacts.github.io/>.
