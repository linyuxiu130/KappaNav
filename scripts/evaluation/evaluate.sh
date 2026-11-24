#!/usr/bin/env bash
# The evaluation standard: all 40 x 100 benchmark tasks (data/benchmark/tasks.json) once, simulation time with a
# 0.2 s decision latency, batch-1 deterministic inference, CUDA graphs, contact stops the robot.
#   bash scripts/evaluation/evaluate.sh --checkpoint MODEL.pt [--output DIR] [--exploration]
# Any task list (Wanderland, NavVerse) with an optional global route (docs/AIRGROUND.md):
#   bash scripts/evaluation/evaluate.sh --benchmark TASKS.json --maps none|oracle|MAP_DIR --checkpoint MODEL.pt
# Writes DIR/summary.json and DIR/tasks.json (the episodes as run); compare two runs paired by episode with
#   python -m kappanav.evaluation.benchmark.results DIR BASELINE_DIR
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common_env.sh"
exec "${KAPPANAV_PYTHON}" -m kappanav.evaluation.benchmark.run "$@"
