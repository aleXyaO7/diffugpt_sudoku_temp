#!/bin/bash

export WANDB_DISABLED=true

module load miniforge/24.3.0-py3.11
conda activate ar_to_diffusion
cd /scratch/qxe3fj/raiselab/ar_to_diffusion/diffugpt_sudoku_temp

python3 -u train_llama3_diffusion-mask_monitoring.optionB.safe.py \
    --load_checkpoint_or_model ./gpt2-model-bs1024-lr1e-3-ep100-20250910-035030/checkpoint-77500 \
    --train_dataset ./data/train \
    --output_dir ./outputs/gpt2-model-to-ar \
    --batch_size 128 \
    --seq_length 164 \
    --max_steps 200 \

