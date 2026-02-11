# CS336 Assignment 1: Basics

My implementation for Stanford CS336 (Spring 2025) Assignment 1. The BPE tokenizer, transformer, optimizers, and training loop are all implemented from scratch.

Full assignment spec: [cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

## What I built

### BPE Tokenizer

Byte-level BPE tokenizer with regex-based pretokenization and special token support. Trained on both TinyStories (10k vocab) and OpenWebText (32k vocab).

- TinyStories compression ratio: ~4.01 bytes/token
- OWT compression ratio: ~4.50 bytes/token
- Throughput: ~700KB/s

### Transformer LM

~17M parameter GPT-2-style model with RMSNorm (pre-norm), RoPE, SwiGLU, and causal multi-head attention. Also implemented ablation variants: no normalization, post-norm, no positional embeddings, and SiLU without gating.

### Training infrastructure

SGD and AdamW optimizers, cosine LR schedule with warmup, gradient clipping, checkpointing, W&B logging, and mixed precision support. Training is fully config-driven via JSON.

### Text generation

Autoregressive decoding with temperature scaling and top-p sampling.

## Project structure

```
cs336_basics/
  bpe_tokenizer.py           # Tokenizer (encode/decode from trained vocab + merges)
  bpe_tokenizer_functions.py  # BPE training algorithm
  transformer_modules.py      # Model components and ablation variants
  training_utils.py           # Optimizers, LR schedules, loss, checkpointing
  decoder.py                  # Text generation
  pretokenization.py          # Chunk boundaries for parallel tokenization
  flops_calculator.py         # FLOPs analysis for GPT-2 variants
  utils.py                    # Benchmarking, sampling

train.py                      # Training script
generate-text.py              # CLI for generation from checkpoints
config.json                   # Training configuration
written_assignment/            # Written analysis and profiling
```

## Setup

Environment is managed with `uv`:

```sh
pip install uv
```

### Tests

All implementations are tested against reference outputs via pytest. Test adapters in `tests/adapters.py` connect the implementations to the test suite.

```sh
uv run pytest
```

### Download data

```sh
mkdir -p data && cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Training

```sh
uv run python train.py config.json
```

Supports model variants (`standard-rope`, `no-norm`, `post-norm`, `nope`, `silu`), mixed precision (`bf16`/`fp16`/`fp32`), and gradient logging. Everything is controlled from the config file.

### Text generation

```sh
uv run python generate-text.py --model-id <uuid> --start-prompt "Once upon a time"
```

## Results

### TinyStories

Best validation loss of 1.32. I used a telescopic search for the learning rate, cosine decay to 10% of max LR with 1% warmup. Best batch size was 64, though the differences were not significant. Larger batch sizes need proportionally larger learning rates.

### OpenWebText

Same architecture and iteration count as TinyStories. Surpassed the naive leaderboard baseline in 25 minutes. After 5000 iterations (~327M tokens), perplexity is around 50. The loss curve looks promising but the model has not learned the finer details of the language yet.

### Ablations

- Removing RMSNorm almost causes divergence during the high LR phase.
- Post-norm performs similarly to pre-norm, slightly worse.
- Removing RoPE gives the worst performance, but frees ~2GB VRAM.
- SiLU without gating is surprisingly competitive at this scale (1.33 vs 1.35 val loss).

### Experiment logs

- [Learning rate tuning](https://wandb.ai/ugurkap/cs336-a1-tuning?nw=nwuserugurkap)
- [Batch size experiments](https://wandb.ai/ugurkap/cs336-a1-batchsize?nw=nwuserugurkap)
- [Ablations](https://wandb.ai/ugurkap/cs336-a1-ablations?nw=nwuserugurkap)
- [OWT Training](https://wandb.ai/ugurkap/cs336-a1-owt?nw=nwuserugurkap)

## Written analysis

Full write-up with FLOPs accounting, memory analysis, and AdamW resource estimation is in [`written_assignment/write_up.md`](./written_assignment/write_up.md).
