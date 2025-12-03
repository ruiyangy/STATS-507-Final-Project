# From Pretraining to Alignment: A Modern Small Language Model

This project implements a modern, decoder-only Transformer language model from scratch, extending the [nanoGPT](https://github.com/karpathy/nanoGPT) framework. It explores the full lifecycle of Large Language Model (LLM) development, from pretraining on educational data to alignment via Supervised Fine-Tuning (SFT).

## 🚀 Key Features

Unlike the standard GPT-2 architecture, this model integrates state-of-the-art primitives found in modern foundation models (e.g., LLaMA):

- **Rotary Positional Embeddings (RoPE):** Replaces absolute embeddings for better context generalization.
- **SwiGLU Activation:** Enhances compute efficiency and performance in feed-forward layers.
- **RMSNorm:** Pre-normalization for stable signal propagation.
- **SFT Pipeline:** Custom data loading and masked loss calculation for instruction tuning.

## 📂 Repository Structure

- **gpt.py**: The core model definition, including the GPTModel, TransformerBlock, and MultiHeadAttention classes.
- **rope.py**: Implementation of Rotary Positional Embeddings.
- **sft.py**: Dataset classes and utilities for SFT, including ChatML formatting and token masking logic.
- **pretrain_gpt.py**: Main training script for the pretraining phase (Next-Token Prediction).
- **sft_gpt.py**: Main training script for the Supervised Fine-Tuning phase.
- **\*.sh**: Shell scripts to launch training jobs on GPU clusters.

## 🛠 Usage

### 1. Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- WandB (for logging)

### 2. Pretraining

To pretrain the base model on the FineWeb-Edu dataset:

```bash
bash pretrain_full.sh
```

### 3. Supervised Fine-Tuning (SFT)

To align the pretrained model using the SmolTalk corpus:

```bash
bash sft_full.sh
```

## 📊 Results

- **Pretraining:** Trained on ~1B tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
- **Alignment:** Fine-tuned on [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) to enable coherent instruction following.

## 🙌 Acknowledgements

- Andrej Karpathy for the original [nanoGPT](https://github.com/karpathy/nanoGPT) codebase.
- Hugging Face for the datasets and tokenizer support.
