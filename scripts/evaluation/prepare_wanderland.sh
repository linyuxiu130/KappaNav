#!/usr/bin/env bash
# Wanderland scenes for CurveNav: the robot-independent geometry (built once) and KAPPANAV_ROBOT's navigation layer,
# then that robot's task list. Raw data: python -m kappanav.evaluation.simulation.wanderland.download.
#   KAPPANAV_ROBOT=scout_mini CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/prepare_wanderland.sh --workers 6
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common_env.sh"
"$KAPPANAV_PYTHON" -m kappanav.evaluation.simulation.wanderland.prepare "$@"
"$KAPPANAV_PYTHON" -m kappanav.evaluation.simulation.wanderland.benchmark --output "$KAPPANAV_WANDERLAND_SCENES/tasks_${KAPPANAV_ROBOT:-dingo}.json"
