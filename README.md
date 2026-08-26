# Kortex

A coding language model trained from scratch on TPU v5e-8 using Kaggle's free tier.

## Architecture

- **Type:** Decoder-only Transformer (GPT-2 style with modern upgrades)
- **Parameters:** ~350M
- **Layers:** 24
- **Heads:** 16 (4 KV heads via Grouped Query Attention)
- **Hidden dim:** 1024
- **FFN dim:** 2816 (SwiGLU activation)
- **Context:** 1024 tokens
- **Norm:** RMSNorm
- **Position:** Rotary Position Embeddings (RoPE)

## Training

- **Hardware:** Kaggle TPU v5e-8 (8 cores, 16GB HBM each)
- **Dataset:** Python code from HuggingFace datasets (streaming)
- **Batch size:** 64 effective (4 per core x 16 gradient accumulation)
- **Precision:** BF16 mixed precision
- **Max session:** 8.5 hours (leaves buffer from Kaggle's 9hr TPU limit)
- **Weekly budget:** 20 hours TPU time

## Features

- Checkpointing every 500 steps + best validation loss
- Auto-resume from latest checkpoint
- Time tracking with auto-stop before session timeout
- Auto-push checkpoints to GitHub
- Auto-push final model to HuggingFace Hub
- Streaming data loading (no disk space wasted)

## How to Use

1. Upload `notebooks/train_kortex.ipynb` to Kaggle
2. Set accelerator to **TPU v5e-8** in notebook settings
3. Add Kaggle secrets: `HF` (HuggingFace token), `WRITE_TK` (GitHub token)
4. Run all cells

## Repository Structure

```
kortex/
├── config/model_config.json   # Model and training config
├── src/
│   ├── model.py               # Transformer architecture
│   ├── tokenizer.py           # BPE code tokenizer
│   ├── dataset.py             # Dataset and data loading
│   ├── trainer.py             # Training loop with checkpointing
│   └── utils.py               # GitHub/HF push, time tracking
├── notebooks/
│   └── train_kortex.ipynb     # Kaggle notebook (run this)
└── checkpoints/               # Auto-created during training
```

## License

MIT
