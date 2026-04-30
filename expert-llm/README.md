# 🏆 Expert LLM — Level 4: Frontier Techniques

## Prerequisites
Complete Baby (L1), Teenage (L2), and Adult (L3) first!

## What's New?

| Concept | What You'll Learn |
|---|---|
| Mixture of Experts | How GPT-4 and Mixtral use sparse models |
| Sliding Window Attention | Mistral's efficient long-context trick |
| Speculative Decoding | 2-3x faster generation with a draft model |
| DPO | Simpler alternative to RLHF |
| Quantization | Shrink models 4x with INT4/INT8 |
| LoRA | Fine-tune billion-param models on a laptop |
| Scaling Laws | How to predict model performance |
| KV-Cache Compression | Paged Attention for serving |
| Tokenizer Deep Dive | SentencePiece and special tokens |

## Files
```
expert-llm/
├── moe.py                    ← Mixture of Experts
├── sliding_window.py         ← Sliding Window Attention
├── speculative_decoding.py   ← Draft model acceleration
├── dpo.py                    ← Direct Preference Optimization
├── quantization.py           ← INT8/INT4 quantization
├── lora.py                   ← Low-Rank Adaptation
├── scaling_laws.py           ← Chinchilla scaling laws
├── run_expert.py             ← Run demos
└── expert_llm_interactive.html
```
