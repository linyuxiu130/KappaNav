#!/usr/bin/env bash
# Whole scenes -> training-ready arrays, with raw recordings retained under ROOT/captures.
# CUDA_VISIBLE_DEVICES=... [WORKERS=N] [HELPERS=N] bash scripts/local_planner/capture_tasks.sh TASKS.json ROOT BUNDLE
# A worker takes the largest unclaimed scene, drives its captures concurrently and labels them, with its CPU-heavy
# geometry in HELPERS helper processes (default 2*cores/workers: helpers idle while their worker renders or compiles).
# Default worker count respects free memory on every selected GPU (5 GiB/worker), a quarter of the CPU cores and scene count.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common_env.sh"
[[ $# -ge 3 ]] || { echo 'usage: capture_tasks.sh TASKS.json ROOT BUNDLE [--runtime FILE]' >&2; exit 2; }
tasks=$1 root=$2 bundle=$3; shift 3
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
slots=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 || $1<m {m=$1} END {print int(m/5120)}')
# nproc honours OMP_NUM_THREADS, which common_env.sh pins to 1 for the solvers; count the usable cores instead.
cores=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
by_memory=$(( slots * ${#devices[@]} )); by_cores=$(( cores >= 8 ? cores / 4 : 1 ))
scenes=$("$KAPPANAV_PYTHON" -c 'import json,sys; print(len({s["case"]["navigation"] for s in json.load(open(sys.argv[1]))["conditions"]}))' "$tasks")
workers=${WORKERS:-$(( by_memory < by_cores ? by_memory : by_cores ))}
(( workers > scenes )) && workers=$scenes
(( workers > 0 )) || { echo 'No capture workers available; select free GPUs and a nonempty task catalog' >&2; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$root/logs"; pids=(); failed=0
for ((w=0;w<workers;w++)); do
  CUDA_VISIBLE_DEVICES=${devices[w%${#devices[@]}]} "$KAPPANAV_PYTHON" -u -m kappanav.local_planner.capture.capture_pipeline --scenarios "$tasks" \
    --root "$root" --bundle "$bundle" "$@" --worker "$w" --workers "$workers" ${HELPERS:+--helpers "$HELPERS"} >> "$root/logs/capture_$w.log" 2>&1 & pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || failed=1; done
(( failed == 0 )) || { echo "capture workers failed; see $root/logs/capture_*.log, then rerun to resume" >&2; exit 1; }
"$KAPPANAV_PYTHON" -m kappanav.local_planner.capture.capture_pipeline --scenarios "$tasks" --root "$root" --bundle "$bundle" "$@" --workers "$workers"
