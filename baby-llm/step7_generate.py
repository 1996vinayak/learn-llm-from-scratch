"""
╔══════════════════════════════════════════════════════════════╗
║           STEP 7: TEXT GENERATION  ✍️                         ║
║           "The Model Writes Its Own Story!"                  ║
╚══════════════════════════════════════════════════════════════╝

🎯 WHAT IS TEXT GENERATION?
============================
After training, our model can GENERATE new text!
It does this one character at a time:

  1. Start with some text (or even just one character)
  2. Model predicts: "What's the most likely NEXT character?"
  3. Add that character to the text
  4. Repeat!

This is called "AUTOREGRESSIVE GENERATION" — each new character
depends on all the previous ones.

📝 EXAMPLE:
  Start:    "the "
  Step 1:   "the " → model predicts 'l' → "the l"
  Step 2:   "the l" → model predicts 'i' → "the li"
  Step 3:   "the li" → model predicts 't' → "the lit"
  Step 4:   "the lit" → model predicts 't' → "the litt"
  Step 5:   "the litt" → model predicts 'l' → "the littl"
  Step 6:   "the littl" → model predicts 'e' → "the little"
  ...and so on!

🌡️ TEMPERATURE:
================
  Temperature controls how "creative" vs "predictable" the model is.

  - Temperature = 0.1 (LOW)  → Very predictable, always picks the top choice
                                Like a careful student who always gives the "safe" answer
  - Temperature = 1.0 (MED)  → Balanced between creative and predictable
                                Like a normal conversation
  - Temperature = 2.0 (HIGH) → Very creative/random, might pick unlikely characters
                                Like a wild poet who makes unexpected choices

  Technically: temperature divides the logits before softmax.
  Lower temperature → sharper probability distribution → less randomness.

🎲 TOP-K SAMPLING:
===================
  Instead of considering ALL possible next characters,
  only consider the top K most likely ones.

  top_k=5 means: "Only choose from the 5 most likely next characters"

  This prevents the model from picking very unlikely characters
  while still allowing some creativity.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()  # Don't compute gradients during generation (faster!)
def generate(model, tokenizer, start_text="The ", max_new_chars=200,
             temperature=0.8, top_k=None):
    """
    Generate new text using our trained model!

    Args:
        model:          Our trained BabyLLM
        tokenizer:      Our BabyTokenizer (to encode/decode)
        start_text:     The beginning of the text (prompt)
        max_new_chars:  How many new characters to generate
        temperature:    Creativity control (0.1=safe, 1.0=normal, 2.0=wild)
        top_k:          Only sample from top K characters (None=all)

    Returns:
        The generated text string
    """

    model.eval()  # Put model in "evaluation mode" (not training)

    # Encode the starting text into numbers
    tokens = tokenizer.encode(start_text)
    tokens = torch.tensor([tokens], dtype=torch.long)  # Shape: (1, len(start_text))

    # Generate one character at a time
    for _ in range(max_new_chars):

        # Crop to the last block_size characters
        # (model can only handle block_size characters at once)
        context = tokens[:, -model.block_size:]

        # Get model predictions
        logits, _ = model(context)

        # We only care about the LAST position's prediction
        # (that's the prediction for the NEXT character)
        logits = logits[:, -1, :]  # Shape: (1, vocab_size)

        # Apply temperature
        # Higher temperature → more uniform distribution → more random
        # Lower temperature → sharper distribution → more deterministic
        logits = logits / temperature

        # Apply top-k filtering (optional)
        if top_k is not None:
            # Find the top-k values
            top_values, _ = torch.topk(logits, top_k)
            # Set everything below the k-th value to -infinity
            logits[logits < top_values[:, [-1]]] = float("-inf")

        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=-1)  # Shape: (1, vocab_size)

        # Sample from the probability distribution
        # (randomly pick a character, weighted by probabilities)
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: (1, 1)

        # Append the new token to our sequence
        tokens = torch.cat([tokens, next_token], dim=1)

    # Decode the full sequence back to text
    generated_text = tokenizer.decode(tokens[0].tolist())

    model.train()  # Put model back in training mode

    return generated_text


def demo_generation(model, tokenizer):
    """
    Show text generation with different settings!
    """

    print("\n" + "=" * 60)
    print("✍️ STEP 7: Generating New Text!")
    print("=" * 60)

    # ─── Generation 1: Normal temperature ───
    print("\n🌡️ Temperature = 0.8 (balanced):")
    print("-" * 40)
    text = generate(model, tokenizer, start_text="The little ",
                    max_new_chars=200, temperature=0.8)
    print(text)

    # ─── Generation 2: Low temperature (more predictable) ───
    print("\n🧊 Temperature = 0.3 (predictable/safe):")
    print("-" * 40)
    text = generate(model, tokenizer, start_text="The little ",
                    max_new_chars=200, temperature=0.3)
    print(text)

    # ─── Generation 3: High temperature (more creative) ───
    print("\n🔥 Temperature = 1.5 (creative/wild):")
    print("-" * 40)
    text = generate(model, tokenizer, start_text="The little ",
                    max_new_chars=200, temperature=1.5)
    print(text)

    # ─── Generation 4: Different starting text ───
    print("\n📝 Different prompt - 'Once upon':")
    print("-" * 40)
    text = generate(model, tokenizer, start_text="Once upon ",
                    max_new_chars=200, temperature=0.8)
    print(text)

    print("\n" + "=" * 60)
    print("🎉 That's our Baby LLM generating text!")
    print("=" * 60)
    print("\n💡 Notice:")
    print("   - Low temperature → repetitive but correct patterns")
    print("   - High temperature → creative but sometimes nonsensical")
    print("   - The model learned patterns from our tiny story!")
    print("   - Real LLMs do the EXACT same thing, just at massive scale!")


# ═══════════════════════════════════════════════════════════
# 🏃 RUN THIS FILE (after training) to generate text
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("⚠️  Run 'run_all.py' to train and generate!")
    print("   This file needs a trained model.")
