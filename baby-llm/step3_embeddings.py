"""
╔══════════════════════════════════════════════════════════════╗
║           STEP 3: EMBEDDINGS  📊                             ║
║           "Giving Characters a Personality"                  ║
╚══════════════════════════════════════════════════════════════╝

🎯 WHAT ARE EMBEDDINGS?
========================
An embedding turns a simple number (like character ID = 5)
into a RICH LIST of numbers that captures MEANING.

Think of it like this:
  - Character ID "5" is just a label (like a name tag)
  - Embedding [0.2, -0.5, 0.8, ...] is a PERSONALITY PROFILE!

🎨 COLOR ANALOGY:
  - "Red" is just a word (like a token ID)
  - But [255, 0, 0] tells you EXACTLY what shade of red (RGB values)
  - Embeddings are like RGB values for words/characters!

📍 POSITIONAL EMBEDDINGS:
========================
The model also needs to know WHERE each character is in the sentence.

  "the cat" → 't' is at position 0, 'h' is at position 1, etc.

Why? Because "cat the" means something different from "the cat"!
ORDER MATTERS in language.

So each character gets TWO embeddings added together:
  1. TOKEN EMBEDDING    → "What character am I?"  (identity)
  2. POSITION EMBEDDING → "Where am I?"           (location)

  Final = Token Embedding + Position Embedding

🔢 EMBEDDING DIMENSION:
========================
  - Each embedding is a list of numbers
  - The LENGTH of this list is called "embedding dimension" (n_embd)
  - GPT-3 uses dimension 12,288 (huge!)
  - Our Baby LLM uses dimension 64 (tiny but works!)
  - Bigger dimension = model can capture more nuance
"""

import torch
import torch.nn as nn


class BabyEmbedding(nn.Module):
    """
    Creates embeddings for our Baby LLM.

    nn.Module is PyTorch's base class for all neural network parts.
    Think of it as a LEGO base plate — we build our pieces on top of it.
    """

    def __init__(self, vocab_size, n_embd, block_size):
        """
        Set up our embedding layers.

        Args:
            vocab_size: How many unique characters we have (~30)
            n_embd:     Size of each embedding vector (64)
            block_size: Maximum sequence length our model can handle (32)
        """
        super().__init__()  # Initialize the parent nn.Module

        # TOKEN EMBEDDING TABLE
        # Think of it as a big lookup table:
        #   Row 0 → embedding for character 0
        #   Row 1 → embedding for character 1
        #   ...etc
        # Shape: (vocab_size, n_embd) → e.g., (30, 64)
        self.token_embedding = nn.Embedding(vocab_size, n_embd)

        # POSITION EMBEDDING TABLE
        # Another lookup table, but for POSITIONS:
        #   Row 0 → embedding for position 0 (first character)
        #   Row 1 → embedding for position 1 (second character)
        #   ...etc
        # Shape: (block_size, n_embd) → e.g., (32, 64)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.block_size = block_size

    def forward(self, token_indices):
        """
        Convert token IDs into rich embedding vectors.

        Args:
            token_indices: Tensor of shape (batch_size, sequence_length)
                          Example: [[2, 7, 4, 1], [5, 3, 8, 2]]
                          (2 sequences, each 4 characters long)

        Returns:
            Tensor of shape (batch_size, sequence_length, n_embd)
            Each character is now a rich vector!

        ┌─────────────────────────────────────────────┐
        │  Input:  [2, 7, 4, 1]  (character IDs)      │
        │                                              │
        │  Token Embed:  [[0.1, 0.3, ...],  ← char 2  │
        │                 [0.5, 0.2, ...],  ← char 7  │
        │                 [0.8, 0.1, ...],  ← char 4  │
        │                 [0.2, 0.7, ...]]  ← char 1  │
        │                                              │
        │  + Position:   [[0.9, 0.1, ...],  ← pos 0   │
        │                 [0.3, 0.4, ...],  ← pos 1   │
        │                 [0.6, 0.8, ...],  ← pos 2   │
        │                 [0.1, 0.5, ...]]  ← pos 3   │
        │                                              │
        │  = FINAL:      [[1.0, 0.4, ...],            │
        │                 [0.8, 0.6, ...],             │
        │                 [1.4, 0.9, ...],             │
        │                 [0.3, 1.2, ...]]             │
        └─────────────────────────────────────────────┘
        """

        B, T = token_indices.shape  # B = batch size, T = sequence length

        # Look up token embeddings
        tok_emb = self.token_embedding(token_indices)  # (B, T, n_embd)

        # Create position indices: [0, 1, 2, ..., T-1]
        positions = torch.arange(T, device=token_indices.device)  # (T,)

        # Look up position embeddings
        pos_emb = self.position_embedding(positions)  # (T, n_embd)

        # ADD them together!
        # Token embedding tells "WHAT character"
        # Position embedding tells "WHERE in the sequence"
        x = tok_emb + pos_emb  # (B, T, n_embd)

        return x


def demo_embeddings():
    """Show how embeddings work with a visual example."""

    print("=" * 60)
    print("📊 STEP 3: Understanding Embeddings")
    print("=" * 60)

    # Small example
    vocab_size = 10   # pretend we have 10 characters
    n_embd = 8        # each embedding has 8 numbers
    block_size = 6    # max sequence length of 6

    embedding = BabyEmbedding(vocab_size, n_embd, block_size)

    # Fake input: batch of 1 sequence, 4 characters long
    # Character IDs: [2, 5, 3, 7]
    fake_input = torch.tensor([[2, 5, 3, 7]])

    print(f"\n📥 Input shape: {fake_input.shape}")
    print(f"   (1 sequence, 4 characters)")
    print(f"   Character IDs: {fake_input[0].tolist()}")

    # Get embeddings
    output = embedding(fake_input)

    print(f"\n📤 Output shape: {output.shape}")
    print(f"   (1 sequence, 4 characters, {n_embd} numbers each)")

    print(f"\n🔍 Embedding for character ID 2 (first character):")
    print(f"   {output[0, 0].detach().numpy().round(3)}")

    print(f"\n🔍 Embedding for character ID 5 (second character):")
    print(f"   {output[0, 1].detach().numpy().round(3)}")

    print(f"\n💡 Notice: Each character is now a RICH vector of {n_embd} numbers!")
    print(f"   These numbers will be LEARNED during training.")
    print(f"   Similar characters will get similar embeddings!")

    print("\n" + "=" * 60)
    print("✅ Embeddings ready! Characters now have 'personalities'.")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════
# 🏃 RUN THIS FILE to see embeddings in action
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo_embeddings()
