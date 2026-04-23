#!/bin/bash
# Generate OpenEvolve configs for all non-interactive algorithmic problems
# and all research problems.
#
# Usage:
#   ./SystemBench/FrontierCS/generate_all_openevolve_configs.sh
#   ./SystemBench/FrontierCS/generate_all_openevolve_configs.sh --model o3
#
# Output:
#   SystemBench/FrontierCS/openevolve_configs/alg_<id>/config.yaml
#   SystemBench/FrontierCS/openevolve_configs/alg_<id>/initial_program.cpp
#   SystemBench/FrontierCS/openevolve_configs/res_<name>/config.yaml
#   SystemBench/FrontierCS/openevolve_configs/res_<name>/initial_program.py

set -euo pipefail
cd "$(dirname "$0")/../.."  # cd to Glia root

MODEL="${1:-o3}"
GENERATOR="SystemBench/FrontierCS/generate_openevolve_config.py"
OUTBASE="SystemBench/FrontierCS/openevolve_configs"

# Non-interactive algorithmic problems (from run_fcs_alg_5problems.sh full list)
ALG_PROBLEMS=(0 1 5 6 7 8 9 11 15 22 23 24 26 27 33 41 42 43 44 45 46 47 48 50 58 61 62 64 72 75 83 87 109 110 111 112 113 121 133 137 138 142 145 147 148 150 151 152 155 156 157 158 159 161 162 163 164 165 166 167 168 169 170 171 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 192 193 205 207 210 211 212 213 214 220 225 227 228 229 239 241 247 248)

echo "Generating OpenEvolve configs (model: $MODEL)"
echo "=============================================="

echo ""
echo "--- Algorithmic problems (${#ALG_PROBLEMS[@]} problems) ---"
for pid in "${ALG_PROBLEMS[@]}"; do
    outdir="${OUTBASE}/alg_${pid}"
    python3 "$GENERATOR" --track algorithmic --problem_id "$pid" --output "$outdir" --model "$MODEL"
done

echo ""
echo "--- Research problems (from problems.txt) ---"
PROBLEMS_FILE="SystemBench/FrontierCS/frontier_cs_repo/research/scripts/problems.txt"
res_count=0
while IFS= read -r problem_id || [[ -n "$problem_id" ]]; do
    # Skip comments and blank lines
    [[ "$problem_id" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${problem_id// }" ]] && continue
    # Sanitize slashes for output directory name
    safe_id="${problem_id//\//__}"
    outdir="${OUTBASE}/res_${safe_id}"
    python3 "$GENERATOR" --track research --problem_id "$problem_id" \
        --output "$outdir" --model "$MODEL"
    res_count=$((res_count + 1))
done < "$PROBLEMS_FILE"

echo ""
echo "Done! Configs written to ${OUTBASE}/"
echo "Total algorithmic: ${#ALG_PROBLEMS[@]}"
echo "Total research: $res_count"
