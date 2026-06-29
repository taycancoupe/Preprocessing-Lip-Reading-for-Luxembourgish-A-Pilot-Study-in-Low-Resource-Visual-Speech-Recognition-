#!/bin/bash
#SBATCH --job-name=tiny_finetune
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00#!/bin/bash
#SBATCH --job-name=tiny_finetune
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/users/mshustrov/BSP/logs/tiny_finetune_%j.out
#SBATCH --error=/mnt/scratch/users/mshustrov/BSP/logs/tiny_finetune_%j.err

cd /mnt/scratch/users/mshustrov/BSP/av_hubert

export PYTHONPATH=/mnt/scratch/users/mshustrov/BSP/av_hubert:/mnt/scratch/users/mshustrov/BSP/av_hubert/avhubert:/mnt/scratch/users/mshustrov/BSP/av_hubert/fairseq:$PYTHONPATH

/mnt/scratch/users/mshustrov/BSP/miniforge3/envs/avhubert/bin/fairseq-hydra-train \
  --config-dir /mnt/scratch/users/mshustrov/BSP/av_hubert/avhubert/conf/finetune \
  --config-name self_large_vox_433h \
  task.data=/mnt/scratch/users/mshustrov/BSP/HPC_project_folder/avhubert_tiny \
  task.label_dir=/mnt/scratch/users/mshustrov/BSP/HPC_project_folder/avhubert_tiny \
  task.tokenizer_bpe_model=/mnt/scratch/users/mshustrov/BSP/tokenizer/lux_spm_unigram1000.model \
  model.w2v_path=/mnt/scratch/users/mshustrov/BSP/checkpoints/large_vox_iter5.pt \
  common.user_dir=/mnt/scratch/users/mshustrov/BSP/av_hubert/avhubert \
  hydra.run.dir=/mnt/scratch/users/mshustrov/BSP/experiments/tiny_finetune \
  distributed_training.distributed_world_size=1 \
  distributed_training.nprocs_per_node=1 \
  dataset.num_workers=0 \
  dataset.max_tokens=300 \
  optimization.max_update=2 \
  lr_scheduler.warmup_steps=0 \
  lr_scheduler.decay_steps=2 \
  model.freeze_finetune_updates=0 \
  checkpoint.save_interval_updates=1 \
  checkpoint.no_epoch_checkpoints=true
