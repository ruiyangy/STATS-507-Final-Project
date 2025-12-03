#!/bin/bash

# Script to run SFT training on the full dataset.
# Designed for larger-scale training on clusters.

echo "Starting SFT training on full dataset..."

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="gpt-sft"
export DATA_PATH="data/"
export MODEL_PATH="models/pretrained-models/"
export TOKENIZERS_PARALLELISM=false

# Use these hyperparameters for your full SFT training
python sft_gpt.py \
    --train_data_path "$DATA_PATH/sft_data_packed.arrow" \
    --val_data_path "$DATA_PATH/smol-smoltalk-dev.jsonl.gz" \
    --model_path "$MODEL_PATH/gpt.1B.1-epoch.model.pth" \
    --context_length 1024 \
    --emb_dim 512 \
    --n_heads 8 \
    --n_layers 12 \
    --drop_rate 0.1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --max_epochs 3 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --output_dir "models/sft-models/" \
    --save_every 1000 \
    --eval_every 500 \
    --wandb_project "gpt-sft-full" \
    --device "auto" \
    --num_workers 4 \
    --seed 42

echo "SFT training on full dataset finished."
