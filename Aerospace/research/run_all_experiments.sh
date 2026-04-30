#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/python"
export MPLCONFIGDIR="$ROOT/results/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

SCENARIOS=(
  baseline
  missed_measurements
  bad_geometry
  lighting_loss
  combined_stress
)

for scenario in "${SCENARIOS[@]}"; do
  python3 "$ROOT/python/scripts/run_experiment.py" \
    --config "$ROOT/experiments/$scenario/config.json"
done

python3 "$ROOT/python/scripts/run_parameter_scan.py" \
  --config "$ROOT/experiments/parameter_scan/config.json"

python3 "$ROOT/python/scripts/analyze_parameter_scan.py" \
  --summary "$ROOT/results/parameter_scan/parameter_scan_summary.csv" \
  --out "$ROOT/results/parameter_scan/analysis"

python3 "$ROOT/python/scripts/analyze_scenarios.py" \
  --results-root "$ROOT/results" \
  --out "$ROOT/results/scenario_analysis"
