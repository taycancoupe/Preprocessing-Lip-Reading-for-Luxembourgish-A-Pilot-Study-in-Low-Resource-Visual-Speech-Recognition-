#!/bin/bash
#SBATCH --job-name=overfit5
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/scratch/users/mshustrov/BSP/logs/overfit5_%j.out
#SBATCH --error=/mnt/scratch/users/mshustrov/BSP/logs/overfit5_%j.err

set -euxo pipefail

log_step () {
  echo
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') :: $1 ====="
}

export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/scratch/users/mshustrov/BSP/av_hubert:/mnt/scratch/users/mshustrov/BSP/av_hubert/avhubert:/mnt/scratch/users/mshustrov/BSP/av_hubert/fairseq:${PYTHONPATH:-}

PY=/mnt/scratch/users/mshustrov/BSP/miniforge3/envs/avhubert/bin/python
FAIRSEQ=/mnt/scratch/users/mshustrov/BSP/miniforge3/envs/avhubert/bin/fairseq-hydra-train

log_step "JOB START"
hostname
date
pwd

log_step "ENTER AV-HUBERT"
cd /mnt/scratch/users/mshustrov/BSP/av_hubert
pwd

log_step "NVIDIA-SMI"
nvidia-smi || true

log_step "PYTHON VERSION"
$PY --version

log_step "TORCH IMPORT / CUDA CHECK"
time $PY -c 'import time; print("before torch", flush=True); t=time.time(); import torch; print("after torch", flush=True); print("import_sec", round(time.time()-t,2), flush=True); print("torch", torch.__version__, flush=True); print("cuda", torch.cuda.is_available(), flush=True); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, flush=True)'

log_step "MANIFEST CHECK"
ls -lah /mnt/scratch/users/mshustrov/BSP/HPC_project_folder/avhubert_overfit5
wc -l /mnt/scratch/users/mshustrov/BSP/HPC_project_folder/avhubert_overfit5/train.tsv
wc -l /mnt/scratch/users/mshustrov/BSP/HPC_project_folder/avhubert_overfit5/train.wrd
wc -l /mnt/scratch/users/mshustrov/BSP/HPC_project_folder/avhubert_overfit5/valid.tsv
wc -l /mnt/scratch/users/mshustrov/BSP/HPC_project_folder/avhubert_overfit5/valid.wrd
wc -l /mnt/scratch/users/mshustrov/BSP/HPC_project_folder/avhubert_overfit5/dict.wrd.txt

log_step "CHECKPOINT CHECK"
ls -lh /mnt/scratch/users/mshustrov/BSP/checkpoints/large_vox_iter5.pt

log_step "START FAIRSEQ TRAINING"
time $FAIRSEQ \
  --config-dir /mnt/scratch/users/mshustrov/BSP/av_hubert/avhubert/conf/finetune \
  --config-name self_large_vox_433h \
  task.data=/mnt/scratch/users/mshustrov/BSP/HPC_project_folder/avhubert_overfit5 \
  task.label_dir=/mnt/scratch/users/mshustrov/BSP/HPC_project_folder/avhubert_overfit5 \
  task.tokenizer_bpe_model=/mnt/scratch/users/mshustrov/BSP/tokenizer/lux_spm_unigram1000.model \
  model.w2v_path=/mnt/scratch/users/mshustrov/BSP/checkpoints/large_vox_iter5.pt \
  common.user_dir=/mnt/scratch/users/mshustrov/BSP/av_hubert/avhubert \
  hydra.run.dir=/mnt/scratch/users/mshustrov/BSP/experiments/overfit5 \
  distributed_training.distributed_world_size=1 \
  distributed_training.nprocs_per_node=1 \
  dataset.num_workers=0 \
  dataset.max_tokens=300 \
  optimization.max_update=500 \
  lr_scheduler.warmup_steps=0 \
  lr_scheduler.decay_steps=2 \
  model.freeze_finetune_updates=0 \
  checkpoint.save_interval_updates=50 \
  checkpoint.no_epoch_checkpoints=true

log_step "TRAINING COMMAND FINISHED"
date
