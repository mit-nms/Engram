#!/bin/bash
# Minimal end-to-end smoke test: verifies the install, that the LLM API is
# reachable, that the agent's shell sandbox spawns, and that a single short
# handoff run completes without errors. Uses cloudcast because it has no
# external data dependencies.
#
# Runs free-of-charge on Groq's free tier by default (open-source Llama 3.3
# 70B). Set GROQ_API_KEY (https://console.groq.com/keys) to use it. If
# OPENAI_API_KEY is set instead, falls back to OpenAI o3.
#
# Usage: bash scripts/smoke_test.sh [model]
#   model defaults to llama-3.3-70b-versatile (Groq) or o3 (OpenAI)

set -eo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${GLIA_RESULTS_BASE_DIR:-$ROOT_DIR/results}/smoke_test"

# Groq exposes an OpenAI-compatible endpoint, so we route through OPENAI_BASE_URL
# and the existing langchain "openai:" code path works unchanged.
if [ -n "${GROQ_API_KEY:-}" ]; then
    export OPENAI_API_KEY="${GROQ_API_KEY}"
    export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
    DEFAULT_MODEL="llama-3.3-70b-versatile"
    PROVIDER="Groq (free tier)"
elif [ -n "${OPENAI_API_KEY:-}" ]; then
    DEFAULT_MODEL="o3"
    PROVIDER="OpenAI"
else
    echo "ERROR: set GROQ_API_KEY (free, https://console.groq.com/keys) or OPENAI_API_KEY." >&2
    exit 1
fi

MODEL="${1:-$DEFAULT_MODEL}"

echo "==============================================="
echo " Smoke test"
echo "  provider:   $PROVIDER"
echo "  model:      $MODEL"
echo "  problem:    cloudcast"
echo "  max_agents: 1"
echo "  agent_timeout: 2 min"
echo "  results:    $RESULTS_DIR"
echo "==============================================="

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
