#!/bin/bash
# train.sh NAME [train args...]: one multi-GPU training in the FOREGROUND (a cloud job ends when its command
# returns). Output /shibo_huang/outputs/globe/NAME, log /shibo_huang/outputs/globe/NAME.log. Rerunning the same NAME
# resumes from its last.pt.
set -o pipefail
NAME=${1:?usage: train.sh NAME [train args]}; shift
cd "$(dirname "$0")/../.." || exit 1
BIN=${PYTHON_BIN:-/opt/conda/envs/lerobot/bin}
OUT=/shibo_huang/outputs/globe/$NAME; LOG=$OUT.log; mkdir -p /shibo_huang/outputs/globe
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
RESUME=; [ -f $OUT/last.pt ] && RESUME=--resume
echo "$(date) $(hostname) $NG GPUs $RESUME $*" | tee -a $LOG
PYTHONPATH=. OMP_NUM_THREADS=4 $BIN/torchrun --standalone --nproc-per-node $NG -m kappanav.global_planner train --output $OUT $RESUME "$@" 2>&1 \
  | grep --line-buffered -v -E "Warning|warnings.warn" | tee -a $LOG
code=$?; echo "$(date) $NAME finished with exit code $code" | tee -a $LOG; exit $code
