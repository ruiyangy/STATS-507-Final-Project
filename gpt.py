"""
GPT Implementation

This file contains all the core classes and functions needed to implement a GPT-style
decoder language model using more recent techniques (e.g., RoPE, SwiGLU, etc.).

"""

import os
import math
import numpy as np
import random
import logging
from typing import Optional, Callable, List, Tuple, Dict, Any

# PyTorch imports
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn import RMSNorm
from torch.amp import autocast, GradScaler

# Data loading imports
from torch.utils.data import Dataset, IterableDataset, TensorDataset, DataLoader
import json
import glob
import gzip
import bz2

# Arrow dataset support
from datasets import load_from_disk

# Tokenization imports
from transformers import AutoTokenizer

# Progress and timing
from tqdm.auto import tqdm, trange
import time

# RoPE imports
from rope import Rotary, apply_rotary_pos_emb


# =============================================================================
# GPT Embedding Layer (with RoPE instead of positional embeddings)
# =============================================================================

class GPTEmbedding(nn.Module):
    """
    GPT Embedding Layer.

    This layer only handles token embeddings. Positional information is handled
    by RoPE in the attention mechanism.
    """
    def __init__(self, vocab_size: int,
                 emb_dim: int = 768,
                 context_length: int = 512):
        """
        Initialize the GPT embedding layer.

        Args:
            vocab_size: Size of the vocabulary
            emb_dim: Embedding dimension
            context_length: Maximum context length (not used in RoPE version)
        """
        super().__init__()

        ###########################################################################
        #                                                                         #
        # 1. Create an embedding layer for tokens (token IDs from the tokenizer). #
        # 2. Note: We don't need positional embeddings since we use RoPE!        #
        ###########################################################################

        # Create token embedding layer
        self.token_embeddings = nn.Embedding(vocab_size, emb_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding layer.

        Args:
            token_ids: Tensor of shape (batch_size, seq_length)
        Returns:
            token embeddings: Tensor of shape (batch_size, seq_length, hidden_dim)
        """
        ###########################################################################
        #                                                                         #
        # 1. Obtain token embeddings from the token embedding layer.              #
        # 2. Return the token embeddings (no positional embeddings needed!)       #
        ###########################################################################

        # Simply return token embeddings and RoPE will handle positional information
        return self.token_embeddings(token_ids)


# =============================================================================
# Multi-Head Attention with RoPE
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Position Embedding (RoPE).

    This implementation uses RoPE to encode positional information directly
    in the attention mechanism instead of using separate positional embeddings.
    """
    def __init__(self, d_in, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initialize Multi-Head Attention with RoPE.

        Args:
            d_in: Dimension of the input embeddings
            context_length: Maximum sequence length (used for attention masking)
            dropout: Dropout probability
            num_heads: Number of attention heads
            qkv_bias: Whether to include bias in Q, K, V projections
        """
        super().__init__()

        ########################################################################################################################
        #                                                                                                                      #
        # 1. Figure out how many dimensions each head should have                                                              #
        # 2. Create linear layers to turn the input embeddings into the query, key, and value projections                      #
        # 3. Calculate the scale factor (1 / sqrt(per-head embedding size))                                                    #
        # 4. Define output projection that merges heads back to model width                                                    #
        # 5. Create dropout module used after attention/MLP projections                                                        #
        # 6. Initialize RoPE for positional encoding                                                                          #
        #                                                                                                                      #
        # NOTE: Each of the Q, K, V projections represents the projections of *each* of the heads as one long sequence.        #
        #       Each of the layers is implicitly representing each head in different parts of its dimensions.                  #
        ######################################################################################################################

        d_out = d_in
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        # Store dimensions
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # Create Q, K, V projection layers
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Output projection to merge heads back
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)


        # Initialize RoPE for positional encoding
        # NOTE: We'll give you this code
        self.rope = Rotary(head_dim=self.head_dim, max_seq_len=context_length, cache_dtypes=(torch.float32,torch.bfloat16))

        
        # Create causal mask: upper triangular matrix with True above diagonal
        # This prevents attending to future positions
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )


    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head attention with RoPE.

        Args:
            embeds: Input embeddings of shape (batch_size, seq_length, d_in)
        Returns:
            Output embeddings of shape (batch_size, seq_length, d_out)
        """
        #################################################################################################################################
        #                                                                                                                               #
        # Implement multi-headed attention with RoPE:                                                                                   #
        #                                                                                                                               #
        # 1. Project input embeddings into Q, K, and V spaces                                                                          #
        # 2. Reshape Q, K, V to separate heads                                                                                         #
        # 3. Apply RoPE to Q and K (this encodes positional information!)                                                              #
        # 4. Compute attention scores: Q @ K^T                                                                                          #
        # 5. Apply causal mask (upper-triangular mask to -inf)                                                                          #
        # 6. Scale attention scores by 1/sqrt(head_dim)                                                                                #
        # 7. Apply softmax to get attention weights                                                                                     #
        # 8. Apply dropout to attention weights                                                                                         #
        # 9. Compute weighted sum: attention_weights @ V                                                                               #
        # 10. Reshape back to original format and apply output projection                                                               #
        #                                                                                                                               #
        # Key insight: RoPE replaces the need for separate positional embeddings!                                                      #
        #################################################################################################################################

        b, num_tokens, d_in = embeds.shape

        # Project to Q, K, V
        queries = self.W_query(embeds)  # (b, num_tokens, d_out)
        keys = self.W_key(embeds)       # (b, num_tokens, d_out)
        values = self.W_value(embeds)   # (b, num_tokens, d_out)
        
        # Reshape to separate heads (b, num_tokens, d_out) to (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose to (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Apply RoPE to queries and keys
        rope_cos, rope_sin = self.rope(queries)
        queries, keys = apply_rotary_pos_emb(queries, keys, rope_cos, rope_sin)

        # Compute attention scores: Q @ K^T
        # (b, num_heads, num_tokens, head_dim) @ (b, num_heads, head_dim, num_tokens)
        attn_scores = queries @ keys.transpose(-2, -1)  # (b, num_heads, num_tokens, num_tokens)
        
        # Apply causal mask which set future positions to -inf
        # Extract mask for current sequence length
        mask_current = self.mask[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_current, float('-inf'))
        
        # Scale attention scores
        attn_scores = attn_scores * self.scale
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum with values
        context = attn_weights @ values
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        
        # Merge heads
        context = context.view(b, num_tokens, self.d_out)
        
        # Apply output projection
        output = self.out_proj(context)
        
        return output


# =============================================================================
# SwiGLU Activation Function
# =============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU activation function with learnable gating mechanism.

    SwiGLU(x) = (xW1) ⊙ Swish(xW2)
    where Swish(x) = x · σ(x)
    """
    def __init__(self, dimension: int):
        """
        Initialize SwiGLU activation.

        Args:
            dimension: Input and output dimension
        """
        super().__init__()
        # NOTE: More recent implementations use a up and down projection for the main and gate paths,
        #       but we'll keep it simple for now.
        self.linear_1 = nn.Linear(dimension, dimension)  # main path
        self.linear_2 = nn.Linear(dimension, dimension)  # gate path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGLU activation.

        Args:
            x: Input tensor of shape [..., dimension]
        Returns:
            Tensor of same shape after SwiGLU gating
        """
        main = self.linear_1(x)
        gate = self.linear_2(x)
        swish_gate = gate * torch.sigmoid(gate)
        return main * swish_gate


# =============================================================================
# Feed-Forward Network
# =============================================================================

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (MLP) used inside a Transformer block.

    This implementation uses the SwiGLU activation function, which is
    computed as: FFN(x) = (Swish(xW1) ⊙ xW2) W3

    (Note: Swish(x) = x * sigmoid(x), which is torch.nn.functional.silu)
    """
    def __init__(self, emb_dim: int, expansion=8/3):
        """
        Initialize the feed-forward network.

        Args:
            emb_dim: Model/embedding width (D)
            expansion: Width multiplier for the hidden layer.
                       Per LLaMA paper, this is typically 2/3 * 4 = 8/3.
        """
        super().__init__()

        ################################################################
        # Implement the layers for the SwiGLU FFN:                     #
        #   1) Calculate the hidden dimension 'd_ff'. This dimension   #
        #      is (expansion * emb_dim).                               #
        #      Reference: LLaMA paper, equation (2).                   #
        #   2) For efficiency, we'll compute W1 and W2 in one step.    #
        #      Define 'self.fc1' as a Linear layer that maps:          #
        #      emb_dim -> 2 * d_ff                                     #
        #   3) Define 'self.fc2' as the output layer that maps:        #
        #      d_ff -> emb_dim                                         #
        ################################################################

        # Calculate the hidden dimension
        # NOTE: The formula for d_ff in SwiGLU is (expansion * emb_dim).
        # We multiply by 2 in the fc1 layer because we are creating
        # *two* matrices (W1 and W2) of size [emb_dim, d_ff].
        d_ff = round(expansion * emb_dim / 2)

        # First layer projects to 2*d_ff to create both main and gate paths
        self.fc1 = nn.Linear(emb_dim, 2 * d_ff)
        # Second layer projects back to original embedding dimension
        self.fc2 = nn.Linear(d_ff, emb_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.

        Args:
            x: Input tensor of shape [..., D]
        Returns:
            Output tensor of shape [..., D]
        """
        ################################################################
        # Implement the forward pass for SwiGLU:                       #
        #   1) Pass the input 'x' through 'self.fc1'.                  #
        #   2) 'Chunk' the result from fc1 into two separate tensors   #
        #      (x1 and x2) along the last dimension.                   #
        #   3) Apply the SwiGLU logic: output = (silu(x1) * x2)        #
        #   4) Pass the result through the output layer 'self.fc2'.    #
        ################################################################

        # Project to 2*d_ff
        x = self.fc1(x)
        
        # Split into two halves: main path and gate path
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply SwiGLU: Swish(gate) * main
        # Swish(x) = x * sigmoid(x) = silu(x)
        x = torch.nn.functional.silu(x1) * x2
        
        # Project back to embedding dimension
        x = self.fc2(x)
        
        return x


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Transformer Block (Decoder Layer) with RoPE.

    This block assembles the core pieces of a GPT-style decoder layer:
    - Multi-head attention with RoPE
    - Position-wise feed-forward network
    - Pre-LayerNorm and residual connections
    """
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize Transformer Block.

        Required cfg keys:
            - emb_dim: int
            - context_length: int
            - n_heads: int
            - n_layers: int
            - drop_rate: float
        """
        super().__init__()

        ################################################################
        # Implement a *decoder-style* Transformer block for GPT with   #
        # pre-norm + residual connections.                             #
        #                                                              #
        # 1) Create a MultiHeadAttention layer with RoPE               #
        # 2) Create the position-wise feed-forward (MLP)               #
        # 3) Create two RMSNorms (pre-norm):                           #
        #      - norm1 applied before attention                        #
        #      - norm2 applied before MLP                              #
        # 4) Store dropout probability; use it after attn and MLP.     #
        ################################################################
        
        # Multi-head attention with RoPE
        self.self_attn = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['drop_rate'],
            num_heads=cfg['n_heads']
        )
        
        # Feed-forward network
        self.ffn = FeedForward(cfg['emb_dim'])
        
        # Layer normalization (pre-norm)
        self.norm1 = RMSNorm(cfg['emb_dim'])
        self.norm2 = RMSNorm(cfg['emb_dim'])
        
        # Dropout probability
        self.dropout_p = cfg['drop_rate']

    def maybe_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout if dropout_p > 0.

        Args:
            x: Input tensor
        Returns:
            Tensor with dropout applied (if enabled)
        """
        ################################################################
        # Apply dropout if dropout_p > 0.                              #
        # - Use nn.functional.dropout(x, p=self.dropout_p,             #
        #   training=self.training)                                    #
        # - Return x unchanged if dropout_p == 0.                      #
        ################################################################
        
        if self.dropout_p > 0:
            return torch.nn.functional.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the transformer block.

        Args:
            x: Input hidden states of shape [B, T, D]
        Returns:
            Output hidden states of shape [B, T, D]
        """
        ################################################################
        # Implement forward pass (pre-norm residual block):            #
        #                                                              #
        # 1. Attention sub-layer (pre-norm + residual):                #
        #    - Apply LayerNorm to input                                #
        #    - Apply MultiHeadAttention with RoPE                      #
        #    - Add residual connection with dropout                    #
        # 2. Feed-forward sub-layer (pre-norm + residual):            #
        #    - Apply LayerNorm to input                                #
        #    - Apply FeedForward network                               #
        #    - Add residual connection with dropout                    #
        ################################################################

        # Attention sub-layer with pre-norm and residual
        # Pre-norm: normalize before attention
        attn_out = self.self_attn(self.norm1(x))
        # Residual connection with dropout
        x = x + self.maybe_dropout(attn_out)
        
        # Feed-forward sub-layer with pre-norm and residual
        # Pre-norm: normalize before FFN
        ffn_out = self.ffn(self.norm2(x))
        # Residual connection with dropout
        x = x + self.maybe_dropout(ffn_out)
        
        return x


# =============================================================================
# GPT Model
# =============================================================================

class GPTModel(nn.Module):
    """
    Complete GPT Model with RoPE.

    This model assembles all components into a unified architecture for
    autoregressive language modeling using RoPE for positional encoding.
    """
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize GPT Model.

        Required cfg keys:
            - vocab_size: int
            - emb_dim: int
            - context_length: int
            - n_heads: int
            - n_layers: int
            - drop_rate: float
        """
        super().__init__()
        self.context_length = cfg['context_length']

        ################################################################
        # Build the GPT model components:                              #
        # 1) Use the embedding layer (token embeddings only)           #
        # 2) Dropout after embedding                                   #
        # 3) Stack of L Transformer blocks (use nn.Sequential)         #
        # 4) Final LayerNorm (pre-logit)                               #
        # 5) Output projection to vocab (nn.Linear(emb_dim, vocab))    #
        # 6) Tie output head weights to input embeddings               #
        #                                                              #
        ################################################################

        # NOTE: Weight tying is when we share the weights between the input embedding
        # and the output head, so there's only one set of weights (fewer parameters).

        # 1. Token embeddings
        self.embedding = GPTEmbedding(
            vocab_size=cfg['vocab_size'],
            emb_dim=cfg['emb_dim'],
            context_length=cfg['context_length']
        )
        
        # 2. Dropout after embedding
        self.dropout = nn.Dropout(cfg['drop_rate'])
        
        # 3. Stack of transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        
        # 4. Final layer normalization
        self.final_norm = RMSNorm(cfg['emb_dim'])
        
        # 5. Output projection to vocabulary
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
        
        # 6. Weight tying: tie output head weights to input embedding weights
        self.out_head.weight = self.embedding.token_embeddings.weight


    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GPT model.

        Args:
            in_idx: Input token IDs of shape [B, T]
        Returns:
            logits: Output logits of shape [B, T, V]
        """
        B, T = in_idx.shape
        if T > self.context_length:
            raise ValueError(f"Sequence length {T} exceeds context_length {self.context_length}")

        ################################################################
        # Forward pass:                                                #
        # 1) Embed the inputs (token embeddings only)                  #
        # 2) Apply dropout                                             #
        # 3) Pass through transformer blocks                           #
        # 4) Apply final LayerNorm                                     #
        # 5) Project to logits via out_head                            #
        # 6) Return logits                                             #
        ################################################################

        # 1. Get token embeddings
        x = self.embedding(in_idx)
        
        # 2. Apply dropout
        x = self.dropout(x)
        
        # 3. Pass through transformer blocks
        x = self.trf_blocks(x)
        
        # 4. Apply final normalization
        x = self.final_norm(x)
        
        # 5. Project to vocabulary logits
        logits = self.out_head(x)
        
        # 6. Return logits
        return logits


# =============================================================================
# Text Generation Functions
# =============================================================================

def generate_new_tokens(model, idx, max_new_tokens, context_size, temperature=1.0):
    """
    Autoregressively generates `max_new_tokens` tokens from the model.

    Args:
        model: The language model
        idx: Starting tensor of shape (batch, seq)
        max_new_tokens: Number of tokens to generate
        context_size: Context window size for the model input
        temperature: Softmax temperature (>0). Lower = more greedy, higher = more random
    Returns:
        idx: The resulting sequence with new tokens appended
    """
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:].to(device)

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # Final token in the sequence
        logits = logits / temperature  # Apply temperature

        probas = torch.softmax(logits, dim=-1)
        # Sample from the distribution rather than argmax for more natural randomness
        idx_next = torch.multinomial(probas, num_samples=1)
        # Keep new token on the same device as the running sequence to avoid device mismatch
        idx_next = idx_next.to(idx.device)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_text(start_context: str, tokenizer, model, max_new_tokens, context_size):
    """
    Generate text from a starting context.

    Args:
        start_context: Starting text prompt
        tokenizer: Tokenizer to use for encoding/decoding
        model: GPT model
        max_new_tokens: Number of tokens to generate
        context_size: Context window size
    Returns:
        Generated text string
    """
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    model.eval()
    out = generate_new_tokens(model=model, idx=encoded_tensor,
                              max_new_tokens=max_new_tokens,
                              context_size=context_size)
    print("Output:", out)
    print("Output length:", len(out[0]))
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    return decoded_text


# =============================================================================
# Dataset Classes
# =============================================================================

class GPTDataset(Dataset):
    """
    Dataset for GPT causal language modeling.

    Creates input/target pairs for next-token prediction by sliding a window
    over tokenized documents.
    """
    def __init__(self, docs: list[str], tokenizer: Any, max_length: int, stride: int):
        """
        Initialize GPT Dataset.

        Args:
            docs: List of raw text documents
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            stride: Step size for sliding window
        """
        ################################################################
        # Goal: Build input/target pairs for next-token prediction.    #
        #                                                              #
        # 1) Store args (tokenizer, max_length, stride).               #
        # 2) Encode the entire text into integer token ids.            #
        # 3) Slide a window of size `max_length` over token_ids with   #
        #    step `stride`. For each start index i:                    #
        #       inputs  = token_ids[i : i + max_length]                #
        #       targets = token_ids[i+1 : i + max_length + 1]          #
        # 4) Keep only full windows; convert to torch.long tensors     #
        #    and append to self.input_ids / self.target_ids.           #
        # Notes: This implements causal LM: predict                    #
        #        token t using tokens < t.                             #
        ################################################################

        # Store parameters
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Initialize lists for input/target pairs
        self.input_ids = []
        self.target_ids = []
        
        # Concatenate all documents into one long text
        full_text = "\n\n".join(docs)
        
        # Encode the entire text into token IDs
        token_ids = tokenizer.encode(full_text)
        
        # Slide a window over the token sequence
        for i in range(0, len(token_ids) - max_length, stride):
            # Input: current window
            input_chunk = token_ids[i : i + max_length]
            # Target: same window shifted by 1 (next token prediction)
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            
            # Only keep full windows
            if len(input_chunk) == max_length and len(target_chunk) == max_length:
                self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
                self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample
        Returns:
            Tuple of (input_ids, target_ids)
        """
        ################################################################
        # Return the input and target tensors for the given index      #
        ################################################################
        
        return self.input_ids[idx], self.target_ids[idx]


class GPTArrowDataset(Dataset):
    """
    Dataset for GPT causal language modeling using pre-packed Arrow datasets.

    This dataset loads pre-processed Arrow datasets where sequences are already
    packed to the maximum length, providing better GPU utilization and faster
    data loading compared to the regular GPTDataset.

    The Arrow dataset should contain:
    - input_ids: List of token IDs (length = max_length)
    - labels: List of target token IDs (length = max_length)
    """
    def __init__(self, arrow_dataset_path: str):
        """
        Initialize GPT Arrow Dataset.

        Args:
            arrow_dataset_path: Path to the Arrow dataset directory
        """
        self.dataset = load_from_disk(arrow_dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample from the Arrow dataset.

        Args:
            idx: Index of the sample
        Returns:
            Tuple of (input_ids, labels) as tensors
        """
        example = self.dataset[idx]
        input_ids = torch.tensor(example['input_ids'], dtype=torch.long)
        labels = torch.tensor(example['labels'], dtype=torch.long)
        return input_ids, labels

# =============================================================================
# DataLoader Creation
# =============================================================================

def create_dataloader(txt=None, arrow_dataset_path=None, batch_size=16, max_length=256, stride=128,
                     shuffle=True, drop_last=True, num_workers=0):
    """
    Create a DataLoader for GPT training.

    This function supports two data formats:
    1. **Raw text format**: List of text documents (txt parameter)
    2. **Arrow dataset format**: Pre-packed Arrow dataset (arrow_dataset_path parameter)

    Args:
        txt: List of text documents (for raw text format)
        arrow_dataset_path: Path to Arrow dataset directory (for Arrow format)
        batch_size: Batch size
        max_length: Maximum sequence length (only used for raw text format)
        stride: Step size for sliding window (only used for raw text format)
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of worker processes
    Returns:
        DataLoader instance
    """
    ################################################################
    # 1) Check if arrow_dataset_path is provided (Arrow format)     #
    # 2) If Arrow format:                                          #
    #    - Create GPTArrowDataset with arrow_dataset_path          #
    #    - Create DataLoader with the Arrow dataset                 #
    # 3) If raw text format:                                       #
    #    - Initialize GPT tokenizer                                #
    #    - Create GPTDataset with txt, tokenizer, max_length, stride#
    #    - Create DataLoader with the regular dataset              #
    # 4) Return the appropriate DataLoader                          #
    ################################################################
    
    # Check if Arrow dataset path is provided
    if arrow_dataset_path is not None:
        # Use pre-packed Arrow dataset
        dataset = GPTArrowDataset(arrow_dataset_path)
    else:
        # Use raw text format
        if txt is None:
            raise ValueError("Either txt or arrow_dataset_path must be provided")
        
        # Initialize tokenizer
        tokenizer = setup_tokenizer()
        
        # Create dataset from raw text
        dataset = GPTDataset(txt, tokenizer, max_length, stride)
    
    # Create and return DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader


# =============================================================================
# Utility Functions
# =============================================================================

def setup_tokenizer():
    """
    Load GPT-2 tokenizer and add special tokens.
    Returns the configured tokenizer.
    """
    ###########################################################################
    #                                                                         #
    # Implement tokenizer setup:                                              #
    #                                                                         #
    # 1. Load GPT-2 tokenizer using AutoTokenizer.from_pretrained()           #
    # 2. Add pad token if missing                                             #
    # 3. Add special tokens for conversations                                 #
    # 4. Test tokenizer with special tokens                                   #
    # 5. Return configured tokenizer                                          #
    #                                                                         #
    # Proper tokenizer setup is crucial for training!                         #
    ###########################################################################

    # NOTE: Use "<|pad|>" as the special token for padding, if needed
    special_tokens_dict = {
        "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    }

    # Load GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    # Add conversation special tokens
    tokenizer.add_special_tokens(special_tokens_dict)
    
    return tokenizer