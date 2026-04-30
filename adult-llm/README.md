# 🎓 Adult LLM — Production-Level Techniques

## Prerequisites
Complete Baby LLM (Level 1) and Teenage LLM (Level 2) first!

## What's New in Level 3?

| Concept | What You'll Learn |
|---|---|
| RoPE | Rotary Position Embeddings (used by LLaMA, GPT-NeoX) |
| Multi-Query Attention | Shared K,V heads for efficiency (used by PaLM, Falcon) |
| SwiGLU | The activation function used by LLaMA and PaLM |
| Flash Attention | Memory-efficient attention algorithm |
| AdamW Deep Dive | Weight decay and optimizer internals |
| Overfitting Detection | Train/Val/Test splits and early stopping |
| Beam Search | Deterministic text generation |
| Mixed Precision | FP16/BF16 training for speed |
| RLHF | How ChatGPT learns from human preferences |

## How to Run
```bash
pip install torch
python run_adult.py
```

## Files
```
adult-llm/
├── rope.py                  ← Rotary Position Embeddings
├── multi_query_attention.py ← MQA / GQA
├── swiglu.py                ← SwiGLU activation
├── adult_model.py           ← Complete model with all techniques
├── beam_search.py           ← Beam search generation
├── train_adult.py           ← Training with mixed precision concepts
├── run_adult.py             ← Run everything
└── adult_llm_interactive.html ← Interactive learning!
```
