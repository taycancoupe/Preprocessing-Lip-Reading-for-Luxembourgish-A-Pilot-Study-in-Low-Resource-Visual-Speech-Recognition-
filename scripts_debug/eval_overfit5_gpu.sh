#!/bin/bash
#SBATCH --job-name=eval_overfit5
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/users/mshustrov/BSP/logs/eval_overfit5_%j.out
#SBATCH --error=/mnt/scratch/users/mshustrov/BSP/logs/eval_overfit5_%j.err

set -euxo pipefail

log_step () {
  echo
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') :: $1 ====="
}

export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/scratch/users/mshustrov/BSP/av_hubert:/mnt/scratch/users/mshustrov/BSP/av_hubert/avhubert:/mnt/scratch/users/mshustrov/BSP/av_hubert/fairseq:${PYTHONPATH:-}

PY=/mnt/scratch/users/mshustrov/BSP/miniforge3/envs/avhubert/bin/python

CKPT=/mnt/scratch/users/mshustrov/BSP/experiments/overfit5/checkpoints/checkpoint_best.pt
DATA=/mnt/scratch/users/mshustrov/BSP/HPC_project_folder/avhubert_overfit5
RESULTS=/mnt/scratch/users/mshustrov/BSP/experiments/overfit5/eval_valid_beam10

log_step "JOB START"
hostname
date
pwd

log_step "NVIDIA-SMI"
nvidia-smi || true

log_step "CHECK FILES"
ls -lh "$CKPT"
ls -lah "$DATA"
mkdir -p "$RESULTS"

log_step "RUN INFERENCE"
cd /mnt/scratch/users/mshustrov/BSP/av_hubert

time $PY /mnt/scratch/users/mshustrov/BSP/av_hubert/avhubert/infer_s2s.py \
  --config-dir /mnt/scratch/users/mshustrov/BSP/av_hubert/avhubert/conf \
  --config-name s2s_decode \
  common.user_dir=/mnt/scratch/users/mshustrov/BSP/av_hubert/avhubert \
  common_eval.path="$CKPT" \
  common_eval.results_path="$RESULTS" \
  dataset.gen_subset=valid \
  dataset.max_tokens=1000 \
  dataset.num_workers=0 \
  override.data="$DATA" \
  override.label_dir="$DATA" \
  override.modalities='["video"]' \
  generation.beam=10

log_step "RESULT FILES"
find "$RESULTS" -maxdepth 1 -type f -exec ls -lh {} \;

log_step "WER"
cat "$RESULTS"/wer.*

log_step "DONE"
date
