"""
SFT Training Script

This script contains the complete training loop for supervised fine-tuning (SFT) of GPT models.

Usage:
    python sft_gpt.py

The script will:
1. Load a pre-trained GPT model
2. Load SFT conversation data
3. Fine-tune the model with masked loss computation
4. Save checkpoints and log to wandb

Need the following components in sft.py:
- SFTDataset: Load and format conversational data with proper token masking
- SFTDatasetFast: Fast tokenization version for better performance
- Data collators: Handle batching for SFT training
- Generation functions: Conversational text generation
- Utility functions: Model loading and validation
"""

import os
import json
import math
import numpy as np
import random
import logging
import argparse
from typing import Optional, Callable, List, Tuple, Dict, Any, Iterable
from copy import deepcopy
import gzip

# PyTorch imports
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn import RMSNorm
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Transformers and tokenization
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, default_data_collator

# Data handling
from datasets import load_from_disk
import orjson

# Progress tracking
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import wandb

# Import our implementations
import gpt
import sft
from gpt import setup_tokenizer as gpt_setup_tokenizer

# Set CuPy/CUDA to allow TF32 computations
# This can provide a speedup on compatible GPUs (RTX 4000 series, etc.)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GPT model with SFT')

    # Data arguments
    parser.add_argument('--train_data_path', type=str,
                       default='/ruiyangy/data/smol-smoltalk-train.jsonl.gz',
                       help='Path to training data')
    parser.add_argument('--train_data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of training data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--val_data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of validation data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--model_path', type=str,
                       default='/ruiyangy/models/pico-gpt/pretrained-models/gpt.1B-18000-step.model.pth',
                       help='Path to pre-trained model')

    # Validation arguments
    parser.add_argument('--val_data_path', type=str,
                       default='/ruiyangy/data/smol-smoltalk-dev.jsonl.gz',
                       help='Path to validation data')
    parser.add_argument('--eval_max_docs', type=int, default=None,
                       help='Maximum number of documents to load for validation (only for raw text)')
    parser.add_argument('--eval_max_docs_step', type=int, default=None,
                       help='Maximum number of validation documents to use during step evaluation (None = use all)')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                       help='Validation batch size')

    # Model arguments
    parser.add_argument('--context_length', type=int, default=1024,
                       help='Context length')
    parser.add_argument('--emb_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=3,
                       help='Maximum number of epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Warmup steps')

    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Evaluation arguments
    parser.add_argument('--eval_every', type=int, default=500,
                       help='Evaluate every N steps')


    # Logging and saving
    parser.add_argument('--output_dir', type=str,
                       default='/ruiyangy/models/pico-gpt/sft-models/',
                       help='Output directory for models')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='Save model every N steps')
    parser.add_argument('--eval_every', type=int, default=1000,
                       help='Evaluate model every N steps')
    parser.add_argument('--wandb_project', type=str, default='gpt-sft',
                       help='Wandb project name')

    # Data arguments
    parser.add_argument('--max_train_docs', type=int, default=None,
                       help='Maximum number of documents to load (for testing)')

    return parser.parse_args()


def setup_device(device_arg):
    """Set up the device for training."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = device_arg

    print(f"Using device: {device}")
    return device


def setup_tokenizer():
    """Set up tokenizer with special tokens for SFT."""
    # Call the setup_tokenizer function from gpt.py
    tokenizer = gpt_setup_tokenizer()

    # Calculate actual vocabulary size
    special_tokens = ["<|user|>", "<|assistant|>", "<|end|>", "<|system|>", "<|pad|>"]
    max_token_id = max(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    actual_vocab_size = max_token_id + 1

    print(f"✅ Tokenizer initialized with {actual_vocab_size} tokens")
    return tokenizer, actual_vocab_size


def create_model_config(args, vocab_size):
    """Create model configuration."""
    config = {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False
    }
    return config


def load_model(model_path, config):
    """Load pre-trained model."""
    print(f"Loading pre-trained model from {model_path}...")
    state_dict = torch.load(model_path, map_location='cpu')

    # Create model with correct configuration
    model = gpt.GPTModel(config)

    # Check vocabulary size compatibility
    original_vocab_size = state_dict['embedding.token_embeddings.weight'].shape[0]
    new_vocab_size = config['vocab_size']

    if original_vocab_size != new_vocab_size:
        print(f"❌ ERROR: Vocabulary size mismatch!")
        print(f"   Model vocab size: {original_vocab_size}")
        print(f"   Expected vocab size: {new_vocab_size}")
        raise ValueError("Vocabulary size mismatch - use the corrected model")
    else:
        print(f"✅ Vocabulary sizes match: {original_vocab_size}")

    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Model loaded successfully!")

    return model


def create_dataloaders(args, tokenizer):
    """Create training and validation dataloaders."""
    print("Creating dataloaders...")

    ###########################################################################
    #                                                                         #
    # Implement SFT DataLoader creation:                                      #
    #                                                                         #
    # 1. Create training DataLoader using sft.create_sft_dataloader():        #
    #    - Use args.train_data_path for data file                             #
    #    - Pass tokenizer, batch_size, context_length                         #
    #    - Set shuffle=True, drop_last=True for training                      #
    #    - Use args.num_workers and args.use_packed                           #
    # 2. Create validation DataLoader using sft.create_sft_dataloader():      #
    #    - Use args.val_data_path for data file                               #
    #    - Pass tokenizer, batch_size, context_length                         #
    #    - Set shuffle=False, drop_last=False for validation                  #
    #    - Use args.num_workers and args.use_packed                           #
    # 3. Print success messages with batch counts                             #
    # 4. Return both dataloaders                                              #
    #                                                                         #
    # This function sets up the data pipeline for SFT training.               #
    ###########################################################################

    # Determine if we should use packed format based on data format
    use_packed_train = (args.train_data_format == 'arrow')
    use_packed_val = (args.val_data_format == 'arrow')
    
    # Create training DataLoader
    train_loader = sft.create_sft_dataloader(
        data_file=args.train_data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.context_length,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        use_packed=use_packed_train
    )
    
    # Create validation DataLoader
    val_loader = sft.create_sft_dataloader(
        data_file=args.val_data_path,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        max_length=args.context_length,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        use_packed=use_packed_val
    )
    
    print(f"Training DataLoader: {len(train_loader)} batches")
    print(f"Validation DataLoader: {len(val_loader)} batches")
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, args, device):
    """Train the model with SFT."""
    print("Starting SFT training...")

    # Move model to device
    model = model.to(device)

    # Loss function - automatically ignores -100 labels
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * args.max_epochs // args.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )

    # Training tracking
    train_losses = []
    val_losses = []
    step = 0
    opt_step = 0
    global_step = 0  # Track total steps across all epochs

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args)
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    model.train()
    last_save_step = -1

    for epoch in trange(args.max_epochs, desc="Epoch"):
        epoch_losses = []

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):

            ###########################################################################
            #                                                                         #
            # Implement SFT forward pass:                                             #
            #                                                                         #
            # 1. Handle different batch formats (dict vs tuple)                      #
            # 2. Move input_ids and labels to the correct device                      #
            # 3. Forward pass with mixed precision:                                   #
            #    - Use torch.amp.autocast() for mixed precision                       #
            #    - Call model(input_ids) to get logits                               #
            #    - Shift logits and labels for next-token prediction                  #
            #    - Compute loss using CrossEntropyLoss with ignore_index=-100         #
            #    - Scale loss by gradient accumulation steps                          #
            #                                                                         #
            # Key insight: SFT uses masked loss where only assistant tokens           #
            # contribute to the loss (labels != -100).                               #
            #                                                                         #
            # Example:                                                                #
            # input_ids: [<|user|>, "Hello", <|end|>, <|assistant|>, "Hi", <|end|>]  #
            # labels:    [-100,     -100,    -100,    50257,        50258, 50259]   #
            # Only tokens 50257, 50258, 50259 contribute to loss!                    #
            ###########################################################################

            # Handle different batch formats (dict vs tuple)
            if isinstance(batch, dict):
                # Packed dataset format: dict with 'input_ids' and 'labels'
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
            else:
                # tuple of (input_ids, labels)
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                # Get model output
                logits = model(input_ids)
                
                # Compute loss
                # Reshape logits (batch, seq_len, vocab_size) to (batch*seq_len, vocab_size)
                # Reshape labels (batch, seq_len) to (batch*seq_len,)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Scale loss by gradient accumulation steps
                loss = loss / args.gradient_accumulation_steps


            ###########################################################################
            #                                                                         #
            # Implement SFT backward pass and optimization:                           #
            #                                                                         #
            # 1. Compute gradients with loss.backward()                               #
            # 2. Update weights only every gradient_accumulation_steps:               #
            #    - Apply gradient clipping with torch.nn.utils.clip_grad_norm_()      #
            #    - Call optimizer.step() to update parameters                         #
            #    - Call scheduler.step() to update learning rate                      #
            #    - Call optimizer.zero_grad() to clear gradients                      #
            #    - Increment opt_step counter                                         #
            # 3. Track loss for logging                                               #
            # 4. Evaluate the model on the validation set every eval_every steps      #
            # 5. Save the model every save_every steps                                #
            # 6. Evaluate the model on the validation set at the end of each epoch    #
            # 7. Save the model at the end of each epoch                              #
            #                                                                         #
            #                                                                         #
            # Example with gradient_accumulation_steps=4:                             #
            # - Steps 1-3: Only compute gradients (no optimizer.step())               #
            # - Step 4: Apply gradients and update parameters                         #
            ###########################################################################

            # Backward pass
            loss.backward()
            
            # Track loss
            epoch_losses.append(loss.item() * args.gradient_accumulation_steps)
            
            # Update step counter
            step += 1
            
            # Update weights only every gradient_accumulation_steps
            if step % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Increment optimizer step counter
                opt_step += 1
                global_step += 1
                
                # Log to wandb
                wandb.log({
                    'train_loss': epoch_losses[-1],
                    'learning_rate': scheduler.get_last_lr()[0],
                    'step': global_step,
                    'epoch': epoch
                })
                
                # Evaluate on validation set
                if global_step % args.eval_every == 0:
                    print(f"\nEvaluating at step {global_step}...")
                    val_loss = sft.evaluate_validation_loss(model, val_loader, loss_fn, device)
                    val_losses.append(val_loss)
                    print(f"Validation loss: {val_loss:.4f}")
                    
                    wandb.log({
                        'val_loss': val_loss,
                        'step': global_step
                    })
                    
                    model.train()
                
                # Save model checkpoint
                if global_step % args.save_every == 0 and global_step != last_save_step:
                    save_path = os.path.join(args.output_dir, f'sft-gpt-step-{global_step}.pth')
                    torch.save(model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
                    last_save_step = global_step

        # End of epoch - evaluate and save
        print(f"\nEpoch {epoch+1} complete!")
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Average training loss: {avg_epoch_loss:.4f}")
        
        # Evaluate on validation set
        print(f"Evaluating at end of epoch {epoch+1}...")
        val_loss = sft.evaluate_validation_loss(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        print(f"Validation loss: {val_loss:.4f}")
        
        wandb.log({
            'epoch_train_loss': avg_epoch_loss,
            'epoch_val_loss': val_loss,
            'epoch': epoch
        })
        
        # Save model at end of epoch
        save_path = os.path.join(args.output_dir, f'sft-gpt-epoch-{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        model.train()

    # Final save
    final_path = os.path.join(args.output_dir, 'sft-gpt-final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

    wandb.finish()
    print("✅ SFT training complete!")


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set up device
    device = setup_device(args.device)

    # Set up tokenizer
    tokenizer, vocab_size = setup_tokenizer()

    # Create model configuration
    config = create_model_config(args, vocab_size)

    ###########################################################################
    #                                                                         #
    # Implement model loading and setup:                                      #
    #                                                                         #
    # 1. Load pre-trained GPT model from checkpoint                           #
    # 2. Move model to the correct device (CPU/GPU)                           #
    # 3. Optionally compile model for better performance                      #
    # 4. Verify model is ready for SFT training                               #
    #                                                                         #
    # This ensures your pre-trained model is properly loaded!                 #
    ###########################################################################

    # Load pre-trained GPT model
    model = load_model(args.model_path, config)
    
    # Move model to device
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Compile model for better performance
    if hasattr(torch, 'compile') and device == 'cuda':
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
            print("Model compiled successfully!")
        except Exception as e:
            print(f"Could not compile model: {e}")
            print("Continuing without compilation...")
    
    # Verify model is ready
    print(f"Model ready for SFT training!")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args, tokenizer)
    
    # Train the model
    train_model(model, train_loader, val_loader, args, device)


if __name__ == "__main__":
    main()
