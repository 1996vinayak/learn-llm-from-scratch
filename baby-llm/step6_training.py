"""
╔══════════════════════════════════════════════════════════════╗
║           STEP 6: TRAINING  🏋️                               ║
║           "Practice Makes Perfect!"                          ║
╚══════════════════════════════════════════════════════════════╝

🎯 WHAT IS TRAINING?
=====================
Training is how the model LEARNS. It's like studying for an exam:

  1. Read a piece of text (input)
  2. Try to predict the next character (guess)
  3. Check if you were right (compare with answer)
  4. Learn from mistakes (adjust weights)
  5. Repeat thousands of times!

🎯 THE TRAINING LOOP:
======================
  ┌─────────────────────────────────────────────┐
  │  for each batch of text:                     │
  │                                              │
  │    1. FORWARD PASS                           │
  │       Feed text into model → get predictions │
  │                                              │
  │    2. COMPUTE LOSS                           │
  │       How wrong were the predictions?        │
  │       (Cross-entropy loss)                   │
  │                                              │
  │    3. BACKWARD PASS (Backpropagation)        │
  │       Calculate: "which weights caused the   │
  │       mistakes?" (compute gradients)         │
  │                                              │
  │    4. UPDATE WEIGHTS                         │
  │       Nudge weights in the right direction   │
  │       (optimizer step)                       │
  │                                              │
  │    5. Repeat!                                │
  └─────────────────────────────────────────────┘

🏀 BASKETBALL ANALOGY:
  1. Shoot the ball (forward pass)
  2. See where it landed — too far left? (compute loss)
  3. Figure out what to adjust — angle? force? (backpropagation)
  4. Adjust your technique slightly (update weights)
  5. Shoot again! (next iteration)

📦 BATCHES:
============
  We don't train on the ENTIRE text at once.
  We take small random chunks called "batches".

  Why?
  - Faster: process small chunks quickly
  - Better learning: randomness helps avoid memorizing
  - Memory: can't fit everything in memory at once

🔧 KEY TRAINING CONCEPTS:
===========================
  - LEARNING RATE: How big each adjustment step is
    Too big → overshoots, never learns
    Too small → learns too slowly
    Just right → steady improvement! (we use 3e-4)

  - EPOCHS: How many times we go through all the data
    More epochs → more practice → better (up to a point)

  - LOSS: A number that measures "how wrong" the model is
    High loss → bad predictions
    Low loss → good predictions
    We want loss to go DOWN over time!
"""

import torch
from step1_data import get_training_data
from step2_tokenizer import BabyTokenizer
from step5_transformer import BabyLLM


def prepare_dataset(text, tokenizer, block_size):
    """
    Prepare the training data.

    We convert the entire text into a tensor of numbers,
    then we'll sample random chunks from it during training.

    Args:
        text:       The training text string
        tokenizer:  Our BabyTokenizer
        block_size: How many characters per training example

    Returns:
        data: A tensor of all the encoded text
    """

    # Encode the entire text into numbers
    encoded = tokenizer.encode(text)

    # Convert to a PyTorch tensor
    data = torch.tensor(encoded, dtype=torch.long)

    return data


def get_batch(data, block_size, batch_size):
    """
    Get a random batch of training examples.

    Each example is a chunk of text:
      - Input (x):  characters [0, 1, 2, ..., block_size-1]
      - Target (y): characters [1, 2, 3, ..., block_size]

    The target is shifted by 1 — because we're predicting the NEXT character!

    Example with block_size=4:
      Text: "the cat sat"
      If we randomly pick starting position 4:
        x = "cat "  (characters at positions 4,5,6,7)
        y = "at s"  (characters at positions 5,6,7,8)

      So the model learns:
        Given 'c', predict 'a'
        Given 'ca', predict 't'
        Given 'cat', predict ' '
        Given 'cat ', predict 's'

    Args:
        data:       The full encoded text tensor
        block_size: Length of each training sequence
        batch_size: How many sequences per batch

    Returns:
        x: Input sequences,  shape (batch_size, block_size)
        y: Target sequences, shape (batch_size, block_size)
    """

    # Pick random starting positions
    # We need room for block_size + 1 characters (input + 1 target)
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))

    # Create input and target sequences
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])

    return x, y


def train_model(
    n_embd=64,
    n_heads=4,
    n_layers=4,
    block_size=32,
    batch_size=16,
    learning_rate=3e-4,
    n_steps=2000,
    print_every=200,
):
    """
    Train our Baby LLM!

    This is where the magic happens. 🪄

    Args:
        n_embd:        Embedding dimension
        n_heads:       Number of attention heads
        n_layers:      Number of transformer blocks
        block_size:    Max sequence length
        batch_size:    Sequences per training step
        learning_rate: How fast to learn (3e-4 is a good default)
        n_steps:       Total training steps
        print_every:   Print progress every N steps
    """

    print("=" * 60)
    print("🏋️ STEP 6: Training Our Baby LLM!")
    print("=" * 60)

    # ─── STEP 1: Prepare Data ───
    print("\n📖 Loading training data...")
    text = get_training_data()
    tokenizer = BabyTokenizer(text)
    data = prepare_dataset(text, tokenizer, block_size)
    print(f"   Text length: {len(data)} characters")
    print(f"   Vocabulary:  {tokenizer.vocab_size} unique characters")

    # ─── STEP 2: Create Model ───
    print("\n🧠 Creating Baby LLM...")
    model = BabyLLM(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_heads=n_heads,
        n_layers=n_layers,
        block_size=block_size,
    )
    n_params = model.count_parameters()
    print(f"   Parameters: {n_params:,}")

    # ─── STEP 3: Create Optimizer ───
    # The optimizer is the "coach" that adjusts the model's weights
    # AdamW is the most popular optimizer for transformers
    # It's like a smart coach that:
    #   - Remembers past adjustments (momentum)
    #   - Adapts learning speed for each weight
    #   - Prevents weights from getting too large (weight decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"   Optimizer: AdamW (learning rate = {learning_rate})")

    # ─── STEP 4: Training Loop! ───
    print(f"\n🚀 Starting training for {n_steps} steps...")
    print(f"   Batch size: {batch_size}")
    print(f"   Block size: {block_size}")
    print("-" * 60)

    losses = []

    for step in range(n_steps):

        # 1. Get a random batch of training data
        x_batch, y_batch = get_batch(data, block_size, batch_size)

        # 2. FORWARD PASS: Feed data through the model
        logits, loss = model(x_batch, y_batch)

        # 3. BACKWARD PASS: Compute gradients
        #    "Which weights caused the mistakes?"
        optimizer.zero_grad(set_to_none=True)  # Clear old gradients
        loss.backward()                         # Compute new gradients

        # 4. UPDATE WEIGHTS: Nudge weights to reduce loss
        optimizer.step()

        # Track and print progress
        losses.append(loss.item())

        if step % print_every == 0 or step == n_steps - 1:
            avg_loss = sum(losses[-print_every:]) / len(losses[-print_every:])
            bar_length = int(30 * (1 - avg_loss / 4))  # Simple progress bar
            bar = "█" * max(bar_length, 1) + "░" * (30 - max(bar_length, 1))
            print(f"   Step {step:5d}/{n_steps} │ Loss: {avg_loss:.4f} │ {bar}")

    print("-" * 60)
    print(f"✅ Training complete!")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   (Started at ~{losses[0]:.4f}, improved by {losses[0] - losses[-1]:.4f})")

    return model, tokenizer


# ═══════════════════════════════════════════════════════════
# 🏃 RUN THIS FILE to train the model
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    model, tokenizer = train_model()
