# 🧑‍🎓 Teenage LLM — Level Up Your Language Model!

## Prerequisites
You should have completed the **Baby LLM** first!
This builds on those concepts with real-world techniques.

## What's New?

| Concept | What You'll Learn |
|---|---|
| BPE Tokenizer | How GPT actually tokenizes text (merge frequent pairs) |
| Dropout | Preventing the model from "memorizing" (overfitting) |
| KV-Cache | Making generation 10x faster |
| LR Scheduling | Warmup + cosine decay (how real models train) |
| Top-p Sampling | Nucleus sampling for better text generation |
| Perplexity | How to measure if your model is actually good |
| Gradient Clipping | Preventing exploding gradients |

## How to Run

```bash
pip install torch
python run_teenage.py
```

## Files
```
teenage-llm/
├── README.md
├── bpe_tokenizer.py        ← BPE tokenizer from scratch
├── advanced_model.py        ← Model with dropout, grad clip, KV-cache
├── lr_scheduler.py          ← Learning rate warmup + cosine decay
├── train_advanced.py        ← Advanced training loop
├── generate_advanced.py     ← Top-p sampling, KV-cache generation
├── evaluate.py              ← Perplexity measurement
├── run_teenage.py           ← Run everything
└── teenage_llm_interactive.html  ← Interactive learning!
```
