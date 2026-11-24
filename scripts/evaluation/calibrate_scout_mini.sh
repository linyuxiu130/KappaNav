#!/usr/bin/env bash
# The Scout Mini's plant in Isaac physics (speed and yaw response, steps, slopes) -> SCOUT_MINI in kappanav.common.physical.
#   bash scripts/evaluation/calibrate_scout_mini.sh --urdf-dir <scout_front_camera> --output scout_mini_calibration.json
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common_env.sh"
export PYTHONPATH="${KAPPANAV_PROJECT_ROOT}:$KAPPANAV_RUNTIME/isaac_torch:$KAPPANAV_RUNTIME/isaac_kit:$KAPPANAV_RUNTIME/navbench_aux"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
exec "$KAPPANAV_ISAAC_PYTHON" -m kappanav.evaluation.simulation.calibrate_scout_mini "$@"
