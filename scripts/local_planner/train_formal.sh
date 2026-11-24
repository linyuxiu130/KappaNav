#!/usr/bin/env bash
# The formal training run of the calibrated policy: from the fresh initial bundle on the evaluation viewpoint
# (outputs/capture/eval_pose), every visible GPU, one BATCH_PER_RANK forward per GPU per step (no accumulation).
# DATASETS adds the supplementary viewpoints (scripts/local_planner/capture_viewpoints.sh) when they are wanted.
#   bash scripts/local_planner/train_formal.sh                (CUDA_VISIBLE_DEVICES selects the GPUs)
#   RESUME=OUTPUT/checkpoints/epoch_NNN bash scripts/local_planner/train_formal.sh   (continue into a new OUTPUT, same GPU count)
# Defaults: DATASETS=outputs/capture/eval_pose, BUNDLE=data/model/initial.pt, OUTPUT=outputs/training/formal_<UTC time>,
# EPOCHS=20, BATCH_PER_RANK=192 (samples per GPU forward: ~180 samples/s per GPU, 71 of 98 GB), global batch
# BATCH_PER_RANK x GPUs (8 GPUs: 1536), LR=1e-3 (tuned at global batch 2560), 1 warmup epoch. The host holds the whole
# dataset (~380 GB of RAM): one training run per machine.
# Extra arguments pass through to kappanav.local_planner.training.bspline_replay. Log: OUTPUT.log.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common_env.sh"
DATASETS=${DATASETS:-outputs/capture/eval_pose}
EPOCHS=${EPOCHS:-20}
BUNDLE=${BUNDLE:-data/model/initial.pt}
OUTPUT=${OUTPUT:-outputs/training/formal_$(date -u +%Y%m%d_%H%M%S)}
BATCH_PER_RANK=${BATCH_PER_RANK:-192}
IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
GLOBAL_BATCH=$(( BATCH_PER_RANK * ${#gpus[@]} ))
LR=${LR:-1e-3}
for dataset in $DATASETS; do
  [[ -f "$dataset/dataset.json" ]] || { echo "$dataset is not a published dataset; finish its capture first" >&2; exit 1; }
done
mkdir -p "$(dirname "$OUTPUT")"
args=(--dataset $DATASETS --output "$OUTPUT" --batch-per-rank "$BATCH_PER_RANK" --global-batch "$GLOBAL_BATCH" --lr "$LR")
[[ -n "${RESUME:-}" ]] || args+=(--epochs "$EPOCHS")
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); else args+=(--bundle "$BUNDLE"); fi
echo "train $(date) datasets=[$DATASETS] output=$OUTPUT gpus=$CUDA_VISIBLE_DEVICES global_batch=$GLOBAL_BATCH lr=$LR" | tee -a "$OUTPUT.log"
bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" "${args[@]}" "$@" 2>&1 | tee -a "$OUTPUT.log"
