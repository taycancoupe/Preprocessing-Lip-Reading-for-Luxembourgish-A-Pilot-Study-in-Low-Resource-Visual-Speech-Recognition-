# Luxembourgish Lip-Reading with AV-HuBERT

This repository contains the reproducible setup for fine-tuning AV-HuBERT on a Luxembourgish lip-reading dataset.

## Contents

- `av_hubert/` - AV-HuBERT source code with local modifications
- `manifests/` - train/valid/test manifests and labels
- `tokenizer/` - Luxembourgish SentencePiece tokenizer
- `scripts/` - SLURM training scripts

## Not included

Large files are not stored in Git:
- pretrained checkpoints
- MP4 clips
- NPZ face crops
- logs
- conda environments

## Current status

The setup currently reaches AV-HuBERT model initialization and dataset loading. The current debugging focus is NPZ video tensor compatibility.
