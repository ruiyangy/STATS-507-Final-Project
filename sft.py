"""
Supervised Fine-Tuning (SFT) Implementation

This file contains all the core classes and functions needed to implement
Supervised Fine-Tuning (SFT) of GPT models for conversational AI.

"""

import os
import json
import math
import numpy as np
import random
import logging
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

# Import GPT components from gpt.py
from gpt import (
    GPTEmbedding,
    MultiHeadAttention,
    SwiGLU,
    FeedForward,
    TransformerBlock,
    GPTModel,
    generate_new_tokens,
    generate_text,
    setup_tokenizer
)


# =============================================================================
# SFT Dataset Class
# =============================================================================

class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning (SFT) of GPT models on conversational data.

    Key Features:
    1. Loads conversations from jsonlines format
    2. Formats with special tokens: <|user|>content<|end|><|assistant|>content<|end|>
    3. SELECTIVE MASKING: Only trains on <|assistant|> token and first <|end|> after assistant
    4. Masks all other tokens including <|user|>, <|system|>, and other <|end|> tokens
    """
    def __init__(self, data_file: str, tokenizer, max_length: int = 1024):
        """
        Initialize SFT Dataset.

        Args:
            data_file: Path to jsonlines file with conversations
            tokenizer: Tokenizer with special tokens added
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations: List[List[Dict[str, str]]] = []

        # Precompute special token IDs
        self.SID = {
            "user": tokenizer.convert_tokens_to_ids("<|user|>"),
            "asst": tokenizer.convert_tokens_to_ids("<|assistant|>"),
            "sys": tokenizer.convert_tokens_to_ids("<|system|>"),
            "end": tokenizer.convert_tokens_to_ids("<|end|>"),
            "pad": tokenizer.pad_token_id or 0,
        }

        ofunc = gzip.open if data_file.endswith(".gz") else open
        with ofunc(data_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
                if not line.strip():
                    continue
                obj = orjson.loads(line)
                msgs = obj.get("messages")
                if isinstance(msgs, list) and msgs:
                    self.conversations.append(msgs)

    def __len__(self):
        return len(self.conversations)

    def _build_ids_labels(self, conversation: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build input IDs and labels for a conversation with proper masking.

        Args:
            conversation: List of message dicts

        Returns:
            Tuple of (input_ids, labels) tensors
        """
        ###########################################################################
        #                                                                         #
        # Implement fast tokenization with selective masking:                    #
        #                                                                         #
        # 1. Initialize empty lists for ids and labels                           #
        # 2. Iterate through each message in the conversation                     #
        # 3. For each message, extract role and content                           #
        # 4. Based on role, append tokens and labels:                             #
        #    - assistant: Add <|assistant|> token (train), content tokens (train), #
        #                first <|end|> token (train)                               #
        #    - user: Add <|user|> token (mask), content tokens (mask),           #
        #            <|end|> token (mask)                                         #
        #    - system: Add <|system|> token (mask), content tokens (mask),       #
        #              <|end|> token (mask)                                        #
        # 5. Use self.SID dictionary for special token IDs                        #
        # 6. Use tokenizer.encode(text, add_special_tokens=False) for content    #
        # 7. Truncate if sequence exceeds max_length                              #
        # 8. Return as torch tensors                                               #
        #                                                                         #
        ###########################################################################

        # Initialize lists for input IDs and labels
        ids = []
        labels = []
        
        # Process each message in the conversation
        for message in conversation:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Tokenize the content (without adding special tokens)
            content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
            
            if role == "assistant":
                # TRAIN on assistant: special token + content + end token
                # Add <|assistant|> token
                ids.append(self.SID["asst"])
                labels.append(self.SID["asst"])
                
                # Add content tokens
                ids.extend(content_tokens)
                labels.extend(content_tokens)
                
                # Add <|end|> token
                ids.append(self.SID["end"])
                labels.append(self.SID["end"])
                
            elif role == "user":
                # MASK user: special token + content + end token
                # Add <|user|> token
                ids.append(self.SID["user"])
                labels.append(-100)
                
                # Add content tokens
                ids.extend(content_tokens)
                labels.extend([-100] * len(content_tokens))
                
                # Add <|end|> token
                ids.append(self.SID["end"])
                labels.append(-100)
                
            elif role == "system":
                # MASK system: special token + content + end token
                # Add <|system|> token
                ids.append(self.SID["sys"])
                labels.append(-100)
                
                # Add content tokens
                ids.extend(content_tokens)
                labels.extend([-100] * len(content_tokens))
                
                # Add <|end|> token
                ids.append(self.SID["end"])
                labels.append(-100)
        
        # Truncate if exceeds max_length
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
            labels = labels[:self.max_length]
        
        # Convert to tensors
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self._build_ids_labels(self.conversations[idx])


# =============================================================================
# Data Collators
# =============================================================================

def sft_data_collator(batch):
    """
    Custom data collator for SFT dataset that handles tuple format.

    Args:
        batch: List of (input_ids, labels) tuples

    Returns:
        Dictionary with batched input_ids and labels
    """
    ###########################################################################
    #                                                                         #
    # Implement SFT data collator for batching:                               #
    #                                                                         #
    # 1. Separate input_ids and labels from batch tuples                      #
    # 2. Find the maximum length in the batch                                #
    # 3. Pad all sequences to the same length:                               #
    #    - Pad input_ids with pad_token_id (usually 0)                       #
    #    - Pad labels with -100 (masked)                                     #
    # 4. Stack into batch tensors                                            #
    # 5. Return dictionary with 'input_ids' and 'labels' keys                #
    #                                                                         #
    # This ensures all sequences in a batch have the same length.            #
    ###########################################################################

    # Separate input_ids and labels from tuples
    input_ids_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    
    # Find maximum length in the batch
    max_len = max(len(ids) for ids in input_ids_list)
    
    # Pad sequences to max_len
    pad_token_id = 0  # Default pad token ID
    
    padded_input_ids = []
    padded_labels = []
    
    for input_ids, labels in zip(input_ids_list, labels_list):
        # Calculate padding needed
        pad_len = max_len - len(input_ids)
        
        # Pad input_ids with pad_token_id
        padded_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        
        # Pad labels with -100; masked, so loss ignores padding
        padded_labs = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
        
        padded_input_ids.append(padded_ids)
        padded_labels.append(padded_labs)
    
    # Stack into batch tensors
    batch_input_ids = torch.stack(padded_input_ids)
    batch_labels = torch.stack(padded_labels)
    
    # Return dictionary
    return {
        'input_ids': batch_input_ids,
        'labels': batch_labels
    }


def hf_collate(examples):
    """
    HuggingFace-style collator for packed datasets.

    This collator is designed for pre-packed Arrow datasets where each example
    already contains a full sequence of input_ids and labels. Unlike regular
    datasets that need padding, packed datasets have sequences that are already
    the correct length (max_length).

    The packed format provides several advantages:
    - No padding needed: sequences are already max_length
    - Better GPU utilization: every token contributes to training
    - Faster data loading: Arrow format is optimized for speed
    - Memory efficiency: supports memory mapping for large datasets

    Args:
        examples: List of examples from packed dataset, each containing:
                 - input_ids: List of token IDs (length = max_length)
                 - labels: List of labels with -100 for masked tokens

    Returns:
        Dictionary with batched data:
        - input_ids: Tensor of shape (batch_size, max_length)
        - labels: Tensor of shape (batch_size, max_length)
        - attention_mask: Tensor indicating non-padding tokens
    """
    pad_id = 0  # Default pad token ID
    ids = torch.tensor(np.stack([e["input_ids"] for e in examples]), dtype=torch.long)
    labs = torch.tensor(np.stack([e["labels"] for e in examples]), dtype=torch.long)
    attn = (ids != pad_id).to(torch.long)
    return {"input_ids": ids, "labels": labs, "attention_mask": attn}


# =============================================================================
# Text Generation Functions
# =============================================================================

def generate_chat_response(model, tokenizer, user_message, max_new_tokens=100, temperature=0.7, context=""):
    """
    Generate a conversational response using the fine-tuned model.

    Args:
        model: Fine-tuned GPT model
        tokenizer: Tokenizer with special tokens
        user_message: User's input message
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)

    Returns:
        Generated response text
    """
    ##############################################################################
    #                                                                            #
    # Implement conversational text generation:                                  #
    #                                                                            #
    # 1. Format input with special tokens: "<|user|>message<|end|><|assistant|>" #
    # 2. Tokenize and move to device                                             #
    # 3. Generate tokens autoregressively:                                       #
    #    - Get model output (logits)                                             #
    #    - Apply temperature scaling and sample next token                       #
    #    - Stop if <|end|> token or max length reached                           #
    # 4. Decode and extract assistant's response                                 #
    #                                                                            #
    # This enables conversational AI by generating responses in chat format.     #
    ##############################################################################

    # Get device from model
    device = next(model.parameters()).device
    
    # Format input with special tokens
    prompt = f"{context}<|user|>{user_message}<|end|><|assistant|>"
    
    # Tokenize and move to device
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Get <|end|> token ID for stopping condition
    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate tokens autoregressively
    generated_ids = input_tensor[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model output
            logits = model(input_tensor)  # (batch, seq_len, vocab_size)
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature
            
            # Convert to probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Add to generated sequence
            generated_ids.append(next_token)
            
            # Stop if <|end|> token generated
            if next_token == end_token_id:
                break
            
            # Update input tensor for next iteration
            input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
    
    # Decode the full generated sequence
    full_response = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    # Extract only the assistant's response (after "<|assistant|>" and before "<|end|>")
    if "<|assistant|>" in full_response:
        # Split by <|assistant|> and take everything after it
        assistant_part = full_response.split("<|assistant|>")[-1]
        # Remove <|end|> token if present
        assistant_response = assistant_part.replace("<|end|>", "").strip()
    else:
        assistant_response = full_response
    
    return assistant_response

def generate_multi_turn_response(model, tokenizer, conversation_history, max_new_tokens=100, temperature=0.7):
    """
    Generate a response considering the full conversation history.

    Args:
        model: Fine-tuned GPT model
        tokenizer: Tokenizer with special tokens
        conversation_history: List of dicts with 'role' and 'content' keys
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text
    """
    # Format the conversation history (excluding the last user message)
    context = ""
    
    # Process all messages EXCEPT the last one to build context
    for message in conversation_history[:-1]:
        role = message['role']
        content = message['content']
        
        if role == 'user':
            context += f"<|user|>{content}<|end|>"
        elif role == 'assistant':
            context += f"<|assistant|>{content}<|end|>"
        elif role == 'system':
            context += f"<|system|>{content}<|end|>"
    
    # The last message should be the user's current query
    last_message = conversation_history[-1]
    if last_message['role'] != 'user':
        raise ValueError("Last message in conversation history must be from user")
    
    user_message = last_message['content']
    
    # Use generate_chat_response with the conversation context
    return generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        user_message=user_message,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        context=context
    )


# =============================================================================
# Utility Functions
# =============================================================================

def load_pretrained_model(model_path: str, config: Dict[str, Any]):
    """
    Load a pre-trained GPT model from checkpoint.

    Args:
        model_path: Path to the model checkpoint
        config: Model configuration dictionary

    Returns:
        Loaded GPT model
    """
    print(f"Loading pre-trained model from {model_path}...")
    state_dict = torch.load(model_path, map_location='cpu')

    # Create model with correct configuration
    model = GPTModel(config)

    # Check if resizing is needed
    original_vocab_size = state_dict['embedding.token_embeddings.weight'].shape[0]
    new_vocab_size = config['vocab_size']

    if original_vocab_size != new_vocab_size:
        print(f"❌ ERROR: Vocabulary size mismatch!")
        print(f"   Model vocab size: {original_vocab_size}")
        print(f"   Expected vocab size: {new_vocab_size}")
        print(f"   Please use the corrected model from the pre-training notebook!")
        raise ValueError("Vocabulary size mismatch - use the corrected model")

    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Model loaded successfully!")
    print(f"✅ Model vocabulary size: {model.embedding.token_embeddings.weight.shape[0]}")

    return model


def evaluate_validation_loss(model, val_loader, loss_fn, device):
    """
    Evaluate the model's loss on the validation dataset.

    Args:
        model: The GPT model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run evaluation on

    Returns:
        Average validation loss
    """
    ###########################################################################
    #                                                                         #
    # Implement validation loss evaluation for SFT:                           #
    #                                                                         #
    # 1. Set model to evaluation mode (model.eval())                          #
    # 2. Initialize loss tracking variables                                   #
    # 3. Iterate through validation batches with torch.no_grad():             #
    #    - Handle different batch formats (dict vs tuple)                     #
    #    - Move data to device                                                #
    #    - Forward pass with mixed precision (optional)                       #
    #    - Compute loss and accumulate                                        #
    # 4. Calculate average validation loss                                    #
    # 5. Set model back to training mode (model.train())                      #
    # 6. Return the average validation loss                                   #
    #                                                                         #
    # This is crucial for monitoring SFT training progress!                   #
    ###########################################################################

    # 1. Set model to evaluation mode
    model.eval()
    
    # 2. Initialize loss tracking
    total_loss = 0.0
    num_batches = 0
    
    # 3. Iterate through validation batches without gradients
    with torch.no_grad():
        for batch in val_loader:
            # Handle different batch formats (dict vs tuple)
            if isinstance(batch, dict):
                # Packed dataset format: dict with 'input_ids' and 'labels'
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
            else:
                # Regular dataset format: tuple of (input_ids, labels)
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Compute loss
            # Reshape logits: (batch, seq_len, vocab_size) -> (batch*seq_len, vocab_size)
            # Reshape labels: (batch, seq_len) -> (batch*seq_len,)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
    
    # 4. Calculate average validation loss
    avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # 5. Set model back to training mode
    model.train()
    
    # 6. Return average validation loss
    return avg_val_loss


def create_sft_dataloader(data_file: str, tokenizer, batch_size: int = 16,
                         max_length: int = 1024, shuffle: bool = True,
                         drop_last: bool = True, num_workers: int = 0,
                         use_packed: bool = False):
    """
    Create a DataLoader for SFT training.

    This function supports two data formats:
    1. **Regular format** (use_packed=False): Individual conversations in jsonlines format
    2. **Packed format** (use_packed=True): Pre-processed Arrow dataset with packed sequences

    Packed datasets are more efficient for training because they:
    - Maximize GPU utilization by filling sequences to max_length
    - Reduce padding overhead
    - Enable faster data loading with Arrow format
    - Support memory mapping for large datasets

    Args:
        data_file: Path to data file (jsonlines or packed dataset)
        tokenizer: Tokenizer with special tokens
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        num_workers: Number of worker processes
        use_packed: Whether to use packed dataset format

    Returns:
        DataLoader instance
    """
    ###########################################################################
    #                                                                         #
    # Implement SFT DataLoader creation:                                      #
    #                                                                         #
    # 1. Check if use_packed is True or False                                 #
    # 2. If use_packed=True:                                                  #
    #    - Use load_from_disk() to load packed dataset                        #
    #    - Create DataLoader with hf_collate function                         #
    # 3. If use_packed=False:                                                 #
    #    - Create SFTDataset instance with data_file, tokenizer, max_length   #
    #    - Create DataLoader with sft_data_collator function                  #
    # 4. Print success message and return DataLoader                          #
    #                                                                         #
    # This provides flexibility to use either regular or packed datasets.     #
    ###########################################################################

    if use_packed:
        # Use packed Arrow dataset format
        print(f"Loading packed SFT dataset from {data_file}...")
        dataset = load_from_disk(data_file)
        
        # Create DataLoader with HuggingFace collator
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=hf_collate
        )
        print(f"Packed SFT DataLoader created with {len(dataset)} examples")
        
    else:
        # Use regular jsonlines format
        print(f"Loading SFT dataset from {data_file}...")
        dataset = SFTDataset(data_file, tokenizer, max_length)
        
        # Create DataLoader with custom SFT collator
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=sft_data_collator
        )
        print(f"SFT DataLoader created with {len(dataset)} conversations")
    
    return dataloader
