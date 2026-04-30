"""
╔══════════════════════════════════════════════════════════════╗
║           🍼 BABY LLM — RUN EVERYTHING! 🚀                   ║
║           "From Zero to Text Generation"                     ║
╚══════════════════════════════════════════════════════════════╝

This script runs ALL the steps in order:
  Step 1: Load and explore training data
  Step 2: Build the tokenizer
  Step 3: (Embeddings are built inside the model)
  Step 4: (Attention is built inside the model)
  Step 5: Build the complete Transformer model
  Step 6: Train the model
  Step 7: Generate new text!

Just run: python run_all.py
"""

import torch
from step1_data import get_training_data, explore_data
from step2_tokenizer import BabyTokenizer
from step6_training import train_model
from step7_generate import generate, demo_generation


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " 🍼 BABY LLM — Building a Language Model From Scratch! ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # ═══════════════════════════════════════════════════════
    # STEP 1: Explore the training data
    # ═══════════════════════════════════════════════════════
    print("\n" + "▶" * 20 + " STEP 1: DATA " + "◀" * 20)
    text = get_training_data()
    explore_data(text)

    # ═══════════════════════════════════════════════════════
    # STEP 2: Build the tokenizer
    # ═══════════════════════════════════════════════════════
    print("\n" + "▶" * 20 + " STEP 2: TOKENIZER " + "◀" * 20)
    tokenizer = BabyTokenizer(text)
    tokenizer.show_vocabulary()
    tokenizer.demo()

    # ═══════════════════════════════════════════════════════
    # STEPS 3-6: Build and train the model
    # (Steps 3, 4, 5 happen inside the model construction)
    # ═══════════════════════════════════════════════════════
    print("\n" + "▶" * 20 + " STEPS 3-6: BUILD & TRAIN " + "◀" * 20)
    model, tokenizer = train_model(
        n_embd=64,        # Embedding size
        n_heads=4,        # Attention heads
        n_layers=4,       # Transformer blocks
        block_size=32,    # Max sequence length
        batch_size=16,    # Sequences per batch
        learning_rate=3e-4,  # Learning speed
        n_steps=2000,     # Training iterations
        print_every=200,  # Print progress interval
    )

    # ═══════════════════════════════════════════════════════
    # STEP 7: Generate new text!
    # ═══════════════════════════════════════════════════════
    print("\n" + "▶" * 20 + " STEP 7: GENERATE! " + "◀" * 20)
    demo_generation(model, tokenizer)

    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📚 WHAT YOU JUST BUILT — A Complete LLM Pipeline!")
    print("=" * 60)
    print("""
    ✅ Step 1: TRAINING DATA
       → Loaded a small story for the model to learn from

    ✅ Step 2: TOKENIZER
       → Built a character-level tokenizer (text ↔ numbers)

    ✅ Step 3: EMBEDDINGS
       → Each character gets a rich vector representation
       → Position embeddings tell the model WHERE each character is

    ✅ Step 4: SELF-ATTENTION
       → Characters can "look at" other characters
       → Multi-head attention captures different patterns
       → Causal mask prevents looking at future characters

    ✅ Step 5: TRANSFORMER
       → Stacked multiple attention + feed-forward blocks
       → Residual connections preserve information
       → Layer normalization keeps training stable

    ✅ Step 6: TRAINING
       → Forward pass: make predictions
       → Loss function: measure how wrong we are
       → Backpropagation: figure out what to adjust
       → Optimizer: update the weights

    ✅ Step 7: TEXT GENERATION
       → Autoregressive: generate one character at a time
       → Temperature: control creativity vs predictability
       → Top-k sampling: limit choices to most likely options

    🎯 These are the EXACT same concepts used in GPT-4, Claude,
       Gemini, and every other modern LLM — just at a much
       larger scale!
    """)

    print("🎉 Congratulations! You've built your own Language Model!")
    print("=" * 60)


if __name__ == "__main__":
    main()
