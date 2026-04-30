#!/bin/bash
# Minimal end-to-end smoke test: verifies the install, that the OpenAI API is
# reachable, that the agent's shell sandbox spawns, and that a single short
# handoff run completes without errors. Uses cloudcast because it has no
# external data dependencies. Expect ~2 minutes wall-clock on o3.
#
# Usage: bash scripts/smoke_test.sh [model]
#   model defaults to o3

set -eo pipefail

MODEL="${1:-o3}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${GLIA_RESULTS_BASE_DIR:-$ROOT_DIR/results}/smoke_test"

echo "==============================================="
echo " Smoke test"
echo "  model:      $MODEL"
echo "  problem:    cloudcast"
echo "  max_agents: 1"
echo "  agent_timeout: 2 min"
echo "  results:    $RESULTS_DIR"
echo "==============================================="

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set." >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: 'docker' is not on PATH; agents require a working docker CLI." >&2
    exit 1
fi

if ! docker ps >/dev/null 2>&1; then
    echo "ERROR: 'docker ps' failed; check that the docker daemon is running and the current user can access it." >&2
    exit 1
fi

export GLIA_RESULTS_BASE_DIR="$RESULTS_DIR"

python "$ROOT_DIR/examples/handoff_example_usage.py" \
    --problem_name cloudcast \
    --model "$MODEL" \
    --num_runs 1 \
    --max_agents 1 \
    --agent_timeout 2 \
    --debug

echo ""
echo "==============================================="
echo "Smoke test finished. Logs under $RESULTS_DIR"
echo "==============================================="
