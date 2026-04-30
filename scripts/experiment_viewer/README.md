# Experiment Results Viewer

Flask backend + HTML/JS frontend to explore Tree, OpenEvolve, and Handoff experiment results. Does **not** require `aggregated_results.json`; uses canonical JSONs and directory layout.

**Requires:** `flask`. Install with `pip install -r scripts/experiment_viewer/requirements.txt` if needed.

## Run

From repo root (so `scripts/` is on Python path):

```bash
cd /path/to/Glia
python scripts/experiment_viewer/server.py
```

Then open http://127.0.0.1:5005 (or the port shown in config).

## Enter results root in the UI

1. Open the app in the browser.
2. In **Results root**, enter one or more directories to scan (one per line). Point roots at run-level folders (parent of each problem folder), not the logs folder itself. For example:
   - `/data2/projects/pantea-work/Glia/results/prompt_ablation_results`
   - `/data2/projects/pantea-work/Glia/results/OpenEvolve_Results`
3. Click **Set & Refresh**. Paths are saved to `config.json` and runs are loaded.
4. Select a run from the dropdown and view summary, score chart, best code, log, and baselines.

Config is stored in `scripts/experiment_viewer/config.json` (`result_roots`, `port`). You can edit that file to set a default port or initial roots.

## Data sources
This viewer loads canonical JSONs by default (no `aggregated_results.json` needed), but if you have
an `aggregated_results.json` file in your problem directory it will be used as a fallback for Tree
and Handoff runs.

- **Tree:** `<problem_folder>/logs/*-deepagents_tree_*rounds.json` (file with **max** round count).
  Iterations have score, node_id, round, iteration; code only in `best_solution`.
- **OpenEvolve:** `checkpoints/` and `logs/openevolve_*.log`; iterations built from `generated_programs/`
  or checkpoints.
- **Handoff:** `<problem_folder>/logs/*-agentic_handoff_*iterations.json` (file with **max** iteration count).

## Features

- **Results root:** Enter paths in the UI; saved to `config.json` and used for discovery.
- Run selector (discovered from roots)
- Summary: best score, total iterations, success count, run type
- Score evolution chart: raw scores + best-so-far envelope (Chart.js)
- Baselines table (when present)
- Best solution code (Prism.js syntax highlighting)
- Log panel (console_output.log or args.json when available)
- Iterations table (first 100)
