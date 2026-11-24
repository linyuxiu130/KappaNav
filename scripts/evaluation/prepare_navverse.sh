#!/usr/bin/env bash
# A NavVerse city for CurveNav: the city geometry (--export, once), KAPPANAV_ROBOT's navigation C-space of the whole
# city, its official tasks ($KAPPANAV_NAVVERSE_SCENES/<city>_<robot>_tasks.json) and, with --long N, N sampled
# long tasks over the --edges shortest-path bins (<city>_<robot>_long_tasks.json).
#   KAPPANAV_ROBOT=scout_mini bash scripts/evaluation/prepare_navverse.sh --scene vcp_amsterdam --export --workers 24 --long 1000
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common_env.sh"
# pxr (USD) comes with the Isaac Sim install; only --export needs it.
usd=${KAPPANAV_USD_LIBS:-/opt/conda/envs/behavior/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311}
export PYTHONPATH="$usd:$PYTHONPATH"
export LD_LIBRARY_PATH="/opt/conda/envs/behavior/lib:$usd/bin:$usd/lib:$LD_LIBRARY_PATH"
exec "$KAPPANAV_PYTHON" -m kappanav.evaluation.simulation.navverse "$@"
