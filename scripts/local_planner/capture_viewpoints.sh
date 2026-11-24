#!/usr/bin/env bash
# The supplementary viewpoints of the formal training set: every task of the calibrated catalog once more, each
# episode's camera at the Dingo mount with base_link nose-up by a pitch drawn uniformly in PITCH_UP degrees (drawn
# from the case id, so independent of workers and resumption). The evaluation viewpoint itself (rest pose, 6 deg
# nose-up = 4 deg down) is outputs/capture/eval_pose.
#   bash scripts/local_planner/capture_viewpoints.sh          (every visible GPU; CUDA_VISIBLE_DEVICES selects)
# Defaults: TASKS=outputs/capture/tasks/tasks.json, ROOT=outputs/capture/pitch_random, BUNDLE=data/model/initial.pt
# (the bundle the evaluation viewpoint was labelled with: one normalization), PITCH_UP="2 9" (1-8 deg down).
# WORKERS/HELPERS as in capture_tasks.sh. Resumable: rerun the same command after an interruption.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common_env.sh"
TASKS=${TASKS:-outputs/capture/tasks/tasks.json}
ROOT=${ROOT:-outputs/capture/pitch_random}
BUNDLE=${BUNDLE:-data/model/initial.pt}
PITCH_UP=${PITCH_UP:-"2 9"}
echo "capture $(date) tasks=$TASKS root=$ROOT pitch_up=[$PITCH_UP] gpus=$CUDA_VISIBLE_DEVICES"
bash "$(dirname "${BASH_SOURCE[0]}")/capture_tasks.sh" "$TASKS" "$ROOT" "$BUNDLE" --pitch-up-degrees $PITCH_UP
echo "done $(date): $ROOT/dataset.json"
