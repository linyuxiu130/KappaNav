#!/usr/bin/env bash
# The one training run: defaults are data/model/initial.pt on data/dataset, 8 epochs, global batch 2560 (160 per GPU
# forward), lr 1e-3, validation every epoch, output outputs/training/<UTC time>. GPUs: CUDA_VISIBLE_DEVICES.
# --resume CHECKPOINT_DIR continues an interrupted run (same GPU count) into a new --output.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common_env.sh"
export NCCL_PROTO="${NCCL_PROTO:-LL}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${KAPPANAV_PYTHON}" -m kappanav.local_planner.training.bspline_replay "$@"
