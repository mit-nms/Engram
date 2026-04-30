import glob
import json
import os
import sys
from pathlib import Path
from statistics import mean, median, pstdev

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from Architect.pricing_table import get_pricing  


def cost_of_row(row: dict) -> float:
    p = get_pricing(row.get("model", ""))
    return (
        row.get("prompt_tokens", 0) / 1_000_000 * p["input"]
        + row.get("completion_tokens", 0) / 1_000_000 * p["output"]
    )


def summarize_run(run_dir: Path) -> dict | None:
    jsonl = run_dir / "openevolve_usage.jsonl"
    if not jsonl.is_file():
        return None
    cost = 0.0
    calls = 0
    prompt_toks = 0
    completion_toks = 0
    reasoning_toks = 0
    max_iter = 0
    models = set()
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cost += cost_of_row(row)
            calls += 1
            prompt_toks += row.get("prompt_tokens", 0)
            completion_toks += row.get("completion_tokens", 0)
            reasoning_toks += row.get("reasoning_tokens", 0) or 0
            if isinstance(row.get("iteration"), int):
                max_iter = max(max_iter, row["iteration"])
            models.add(row.get("model", "unknown"))
    return {
        "run": run_dir.name,
        "cost": cost,
        "calls": calls,
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion_toks,
        "reasoning_tokens": reasoning_toks,
        "max_iter": max_iter,
        "models": sorted(models),
    }


def expand(args: list[str]) -> list[Path]:
    out: list[Path] = []
    for a in args:
        matches = glob.glob(a)
        if matches:
            out.extend(Path(m) for m in matches)
        else:
            out.append(Path(a))
    return [p for p in out if p.is_dir()]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    dirs = sorted(expand(argv))
    rows: list[dict] = []
    for d in dirs:
        r = summarize_run(d)
        if r is not None:
            rows.append(r)
    if not rows:
        print("no openevolve_usage.jsonl found in any dir", file=sys.stderr)
        return 1

    print(f"{'run':<55} {'iter':>5} {'calls':>6} {'cost ($)':>10}")
    print("-" * 80)
    for r in rows:
        print(f"{r['run']:<55} {r['max_iter']:>5} {r['calls']:>6} {r['cost']:>10.2f}")
    print("-" * 80)

    costs = [r["cost"] for r in rows]
    print(f"{'n runs':<55} {'':>5} {'':>6} {len(rows):>10}")
    print(f"{'total':<55} {'':>5} {'':>6} {sum(costs):>10.2f}")
    print(f"{'mean':<55} {'':>5} {'':>6} {mean(costs):>10.2f}")
    print(f"{'median':<55} {'':>5} {'':>6} {median(costs):>10.2f}")
    if len(costs) > 1:
        print(f"{'stdev':<55} {'':>5} {'':>6} {pstdev(costs):>10.2f}")
        print(f"{'min/max':<55} {'':>5} {'':>6} {min(costs):>5.2f}/{max(costs):<.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
