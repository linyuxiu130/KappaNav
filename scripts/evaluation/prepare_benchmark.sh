#!/usr/bin/env bash
# The benchmark, in one pass: Habitat scene assets from the official USD scenes (if absent), the navigation geometry
# of the 40 evaluation scenes (WORKERS_PER_GPU processes on each visible GPU), then the task set ROOT/tasks.json from
# the imported official pairs ROOT/pointgoal.
#   bash scripts/evaluation/prepare_benchmark.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common_env.sh"
ROOT=${ROOT:-data/benchmark}; WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
if [[ ! -d "$ROOT/assets" ]]; then
  # USD and trimesh come from the Isaac evaluation environment.
  USD=$(echo "$(dirname "$(dirname "$KAPPANAV_ISAAC_PYTHON")")"/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*)
  PYTHONPATH="$USD:$PYTHONPATH" LD_LIBRARY_PATH="$USD/bin:$LD_LIBRARY_PATH" "$KAPPANAV_ISAAC_PYTHON" \
    -m kappanav.evaluation.benchmark.assets --source "$KAPPANAV_ASSETS/evaluation/scenes" --output "$ROOT/assets" --workers 14
fi
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"; workers=$(( ${#devices[@]} * WORKERS_PER_GPU ))
mkdir -p "$ROOT/logs"; pids=(); failed=0
for ((w=0;w<workers;w++)); do
  CUDA_VISIBLE_DEVICES=${devices[w%${#devices[@]}]} "$KAPPANAV_PYTHON" -u -m kappanav.evaluation.benchmark.prepare --root "$ROOT" \
    --worker "$w" --workers "$workers" > "$ROOT/logs/geometry_${w}.log" 2>&1 & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "geometry failed; see $ROOT/logs" >&2; exit 1; }
"$KAPPANAV_PYTHON" -m kappanav.evaluation.benchmark.prepare --root "$ROOT"
