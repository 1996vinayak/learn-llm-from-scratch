# 🍼 Baby LLM — Build Your Own Language Model From Scratch

## What Are We Building?

A **tiny language model** that learns to write text, character by character.
We train it on a small story, and then it generates NEW text that looks similar!

---

## 🧠 What is an LLM (Large Language Model)?

Imagine a **super smart parrot** 🦜:
- It listens to LOTS of sentences
- It learns patterns like "after 'hel' usually comes 'lo'"
- Then it can **predict** what comes next!

That's basically what ChatGPT, Gemini, etc. do — but with BILLIONS of patterns.
We'll do the same thing, but **tiny** (a Baby LLM! 🍼)

---

## 🏗️ The Building Blocks (What Every LLM Has)

```
┌─────────────────────────────────────────────────┐
│                  OUR BABY LLM                    │
│                                                  │
│  1. 📖 TRAINING DATA     → The text it reads    │
│  2. 🔤 TOKENIZER         → Breaks text into     │
│                             pieces (characters)  │
│  3. 📊 EMBEDDINGS        → Turns characters      │
│                             into numbers         │
│  4. 🧠 TRANSFORMER       → The "brain" that      │
│                             learns patterns      │
│     ├── Self-Attention   → "What's important?"   │
│     ├── Feed-Forward     → "Think about it"      │
│     └── Layer Norm       → "Stay balanced"       │
│  5. 🎯 TRAINING LOOP     → Practice makes        │
│                             perfect!             │
│  6. ✍️  TEXT GENERATION   → Write new text!       │
└─────────────────────────────────────────────────┘
```

---

## 📁 File Structure

```
baby-llm/
├── README.md              ← You are here! 📍
├── requirements.txt       ← Libraries we need
├── step1_data.py          ← Step 1: Prepare training data
├── step2_tokenizer.py     ← Step 2: Build a tokenizer
├── step3_embeddings.py    ← Step 3: Turn text into numbers
├── step4_attention.py     ← Step 4: Build Self-Attention
├── step5_transformer.py   ← Step 5: Build the Transformer
├── step6_training.py      ← Step 6: Train the model
├── step7_generate.py      ← Step 7: Generate new text!
└── run_all.py             ← Run everything at once! 🚀
```

---

## 🚀 How to Run

```bash
# 1. Install Python (3.8 or higher)
# 2. Install the library we need:
pip install torch

# 3. Run the full Baby LLM:
python run_all.py
```

---

## 📚 The Analogy We'll Use Throughout

Think of our LLM like a **cooking student** 👨‍🍳:

| LLM Concept      | Cooking Analogy                              |
|-------------------|----------------------------------------------|
| Training Data     | Recipe book 📖                               |
| Tokenizer         | Cutting ingredients into pieces 🔪           |
| Embeddings        | Giving each ingredient a nutrition label 🏷️  |
| Self-Attention    | Knowing which ingredients go together 🤝     |
| Feed-Forward      | Actually mixing and cooking 🍳               |
| Training          | Practicing the recipe many times 🔄          |
| Generation        | Cooking a NEW dish from memory! 🍽️          |

---

## ⚡ Important Note

This is a TINY model for LEARNING. Real LLMs like GPT-4 have:
- Billions of parameters (ours has ~thousands)
- Trained on the entire internet (ours uses a small story)
- Run on massive GPU clusters (ours runs on your laptop!)

But the CONCEPTS are exactly the same! 🎯
