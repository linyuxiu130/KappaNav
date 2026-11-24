#!/usr/bin/env bash
# The benchmark in Isaac Sim: same tasks, decision delay, controller and summary as evaluate.sh.
#   bash scripts/evaluation/evaluate_isaac.sh --checkpoint data/model/policy.pt --workers 8
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common_env.sh"
export PYTHONPATH="${KAPPANAV_PROJECT_ROOT}:$KAPPANAV_RUNTIME/isaac_torch:$KAPPANAV_RUNTIME/isaac_kit:$KAPPANAV_RUNTIME/navbench_aux:${ACADOS_SOURCE_DIR}/interfaces/acados_template"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "$KAPPANAV_ISAAC_PYTHON" -m kappanav.evaluation.benchmark.isaac --scenes "$KAPPANAV_ASSETS/evaluation/scenes" \
  --robot "$KAPPANAV_ASSETS/evaluation/robots/dingo.usd" "$@"
