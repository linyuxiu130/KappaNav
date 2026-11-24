# Shared environment of every script. The machine-specific locations below can be overridden from the environment.
KAPPANAV_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAPPANAV_PYTHON=${KAPPANAV_PYTHON:-/shibo_huang/envs/curvenav-pcard/bin/python}          # Habitat, training
KAPPANAV_ISAAC_PYTHON=${KAPPANAV_ISAAC_PYTHON:-/shibo_huang/envs/curvenav-eval-5.0/bin/python}  # Isaac Sim 5.0
KAPPANAV_RUNTIME=${KAPPANAV_RUNTIME:-/shibo_huang/data/curvenav/runtime}                  # see data/README.md
KAPPANAV_ASSETS=${KAPPANAV_ASSETS:-/shibo_huang/data/curvenav/assets}                     # source scenes and robot
KAPPANAV_HSSD=${KAPPANAV_HSSD:-/shibo_huang/datasets/hssd-hab}                       # HSSD training scenes
export KAPPANAV_WANDERLAND=${KAPPANAV_WANDERLAND:-/shibo_huang/data/wanderland}       # raw/ and the public manifest
export KAPPANAV_WANDERLAND_SCENES=${KAPPANAV_WANDERLAND_SCENES:-$KAPPANAV_WANDERLAND/curvenav}  # processed scenes
export KAPPANAV_NAVVERSE=${KAPPANAV_NAVVERSE:-/shibo_huang/data/navverse/release}          # the NavVerse release
export KAPPANAV_NAVVERSE_SCENES=${KAPPANAV_NAVVERSE_SCENES:-/shibo_huang/data/navverse/curvenav}  # processed cities
export KAPPANAV_BASELINE_PYTHON=${KAPPANAV_BASELINE_PYTHON:-/shibo_huang/envs/curvenav-baselines/bin/python}  # baseline servers
export KAPPANAV_BASELINE_WEIGHTS=${KAPPANAV_BASELINE_WEIGHTS:-/shibo_huang/data/curvenav/weights}  # their released checkpoints
# KAPPANAV_ROBOT selects the robot of the whole run (kappanav.common.physical): dingo (default) or scout_mini.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
export CUDA_VISIBLE_DEVICES
export ACADOS_SOURCE_DIR="$KAPPANAV_RUNTIME/acados"
export PYTHONPATH="${KAPPANAV_PROJECT_ROOT}:$KAPPANAV_RUNTIME/habitat_dingo_0_3_3:$KAPPANAV_RUNTIME/navbench_aux:${ACADOS_SOURCE_DIR}/interfaces/acados_template${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/shibo_huang/data/curvenav/cache}
cd "${KAPPANAV_PROJECT_ROOT}"
