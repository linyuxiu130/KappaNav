#!/usr/bin/env bash
# The one data pipeline: training scene geometry -> task catalog -> scene workers publish the training set.
#   bash scripts/local_planner/prepare_data.sh            (CUDA_VISIBLE_DEVICES selects the GPUs)
# Writes OUT/scenes/{NAME/navigation,tasks_NAME.json,tasks.json} and OUT/dataset/{captures,shards,train,validation,dataset.json}.
# Defaults (the current training set): OUT=data, SOURCES="grs=GRScenes hssd=HSSD" asset roots below, BUNDLE=OUT/model/initial.pt
# (records the candidate banks and fixes the array normalization), TASKS_PER_SCENE=20, WALL_TASKS_PER_SCENE=10 (escape starts
# at or near obstacles), EXCLUDE = the degenerate GRScenes scene and the HSSD scene with unresolved assets.
# PITCH_UP="LOW HIGH" (default "2 9"): each episode's camera is the Dingo mount with base_link nose-up by a pitch drawn
# uniformly in [LOW, HIGH] degrees (rest pose: 6; PITCH_UP= keeps the runtime camera).
# WORKERS (capture workers over the GPUs) and HELPERS (CPU helpers per worker) default from cores and GPU memory, see capture_tasks.sh.
# Every stage resumes: rerun the same command after an interruption. Changed inputs need a new OUT.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common_env.sh"
OUT=${OUT:-data}
SOURCES=${SOURCES:-"grs=$KAPPANAV_ASSETS/training/grscenes-hab hssd=$KAPPANAV_HSSD"}
BUNDLE=${BUNDLE:-$OUT/model/initial.pt}
EXCLUDE=${EXCLUDE-"MWAX5JYKTKJZ2AABAAAAAAQ8_usd 105515175_173104107"}
SCENES="$OUT/scenes" ROOT="$OUT/dataset"
sources=(); for s in $SOURCES; do sources+=(--source "$s"); done
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
mkdir -p "$SCENES/logs" "$ROOT/logs"

run() {   # run NAME WORKERS COMMAND...: WORKERS processes over the GPUs, each given --worker/--workers
  local name=$1 workers=$2; shift 2; local pids=() failed=0
  for ((w=0;w<workers;w++)); do
    CUDA_VISIBLE_DEVICES=${devices[w%${#devices[@]}]} "$KAPPANAV_PYTHON" -u "$@" --worker "$w" --workers "$workers" \
      > "$SCENES/logs/${name}_${w}.log" 2>&1 & pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  (( failed == 0 )) || { echo "$name failed; see $SCENES/logs/${name}_*.log" >&2; exit 1; }
}

echo "geometry $(date)"
run geometry ${#devices[@]} -m kappanav.local_planner.capture.prepare_scenes "${sources[@]}" --output "$SCENES" --exclude $EXCLUDE
echo "tasks $(date)"
[[ -f "$SCENES/tasks.json" ]] || "$KAPPANAV_PYTHON" -m kappanav.local_planner.capture.prepare_scenes "${sources[@]}" --output "$SCENES" --exclude $EXCLUDE \
  --tasks-per-scene "${TASKS_PER_SCENE:-20}" --wall-tasks-per-scene "${WALL_TASKS_PER_SCENE:-10}"
echo "captures $(date)"
PITCH_UP=${PITCH_UP-"2 9"}
bash "$(dirname "${BASH_SOURCE[0]}")/capture_tasks.sh" "$SCENES/tasks.json" "$ROOT" "$BUNDLE" ${PITCH_UP:+--pitch-up-degrees $PITCH_UP}
echo "done $(date): $ROOT/dataset.json"
