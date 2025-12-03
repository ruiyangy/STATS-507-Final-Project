#!/bin/bash

# Full GPT Training Script
# This script trains the GPT model with production-ready hyperparameters
# Designed for use on Great Lakes cluster with GPU resources

echo "Starting full GPT training..."
echo "=================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="gpt-pretraining"
export DATA_PATH="data/"
export OUTPUT_DIR="models/pretrained-models/"
export TOKENIZERS_PARALLELISM=false

# Use these hyperparameters for your full pretraining

# Training hyperparameters for full model
python pretrain_gpt.py \
    --batch_size 16 \
    --learning_rate 6e-4 \
    --max_epochs 15 \
    --emb_dim 512 \
    --n_layers 12 \
    --n_heads 8 \
    --context_length 1024 \
    --save_every 1000 \
    --eval_every 500 \
    --device cuda \
    --data_path $DATA_PATH/fineweb-edu-sample-1B-hf/ \
    --data_format arrow \
    --output_dir $MODEL_PATH \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "gpt-pretraining-$(date +%Y%m%d-%H%M%S)"

echo "Training completed!"
echo "Check the output directory for saved models and logs."
