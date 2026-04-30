"""
╔══════════════════════════════════════════════════════════════╗
║           STEP 5: THE TRANSFORMER  🧠                        ║
║           "Putting It All Together — The Brain!"             ║
╚══════════════════════════════════════════════════════════════╝

🎯 WHAT IS A TRANSFORMER?
===========================
The Transformer is the ARCHITECTURE (blueprint) that powers ALL modern LLMs.
It was invented in 2017 in the famous paper "Attention Is All You Need".

Think of it as a FACTORY with multiple identical floors (layers):

  ┌─────────────────────────────────────────────┐
  │              THE TRANSFORMER                 │
  │                                              │
  │  Input Text                                  │
  │      ↓                                       │
  │  [Embeddings] ← Step 3                       │
  │      ↓                                       │
  │  ┌──────────────────────┐                    │
  │  │  TRANSFORMER BLOCK 1 │ ← One "floor"      │
  │  │  • Self-Attention     │   of the factory   │
  │  │  • Feed-Forward       │                    │
  │  │  • Layer Norm         │                    │
  │  └──────────────────────┘                    │
  │      ↓                                       │
  │  ┌──────────────────────┐                    │
  │  │  TRANSFORMER BLOCK 2 │ ← Another floor     │
  │  │  • Self-Attention     │   (same design!)   │
  │  │  • Feed-Forward       │                    │
  │  │  • Layer Norm         │                    │
  │  └──────────────────────┘                    │
  │      ↓                                       │
  │  [Output Layer] → Predict next character!     │
  └─────────────────────────────────────────────┘

🏗️ COMPONENTS OF EACH BLOCK:
==============================

  1. SELF-ATTENTION (Step 4)
     → "Look at other characters and gather information"
     → Like a group discussion

  2. FEED-FORWARD NETWORK
     → "Think about what you gathered"
     → Like individual study time after the discussion
     → It's just: Linear → ReLU → Linear
       (expand the data, apply non-linearity, compress back)

  3. LAYER NORMALIZATION
     → "Keep the numbers in a reasonable range"
     → Like a teacher saying "everyone calm down!"
     → Prevents numbers from getting too big or too small

  4. RESIDUAL CONNECTIONS (Skip Connections)
     → "Don't forget what you already knew!"
     → output = input + processed(input)
     → Like taking notes: you keep your old notes AND add new ones

🤔 WHY MULTIPLE BLOCKS?
========================
  Each block learns DIFFERENT patterns:
  - Block 1 might learn: "vowels often follow consonants"
  - Block 2 might learn: "words tend to repeat in this text"

  More blocks = deeper understanding (but slower training)
  GPT-3 has 96 blocks! We'll use 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from step3_embeddings import BabyEmbedding
from step4_attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    The Feed-Forward Network — "Individual thinking time"

    After attention gathers information from other characters,
    the feed-forward network PROCESSES that information.

    It's a simple 2-layer neural network:
      Input → Expand (×4) → ReLU → Compress (÷4) → Output

    🍔 BURGER ANALOGY:
      - Input: flat patty
      - Expand: smash it wide (more surface area = more processing)
      - ReLU: grill it (non-linear transformation!)
      - Compress: shape it back to original size
      - Output: cooked patty!
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # Expand: n_embd → 4 * n_embd
            # (64 → 256) — gives the network more room to think!
            nn.Linear(n_embd, 4 * n_embd),

            # ReLU activation: if x > 0, keep it; if x < 0, make it 0
            # This is the "non-linearity" that lets neural networks
            # learn complex patterns (without it, everything is just
            # straight lines!)
            nn.ReLU(),

            # Compress back: 4 * n_embd → n_embd
            # (256 → 64) — back to original size
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    ONE Transformer Block — one "floor" of our factory.

    Each block does:
    1. Self-Attention (with residual connection + layer norm)
    2. Feed-Forward   (with residual connection + layer norm)

    ┌─────────────────────────────────────────┐
    │         TRANSFORMER BLOCK               │
    │                                          │
    │  Input ──────────────────┐               │
    │    ↓                     │ (residual)    │
    │  LayerNorm               │               │
    │    ↓                     │               │
    │  Self-Attention          │               │
    │    ↓                     │               │
    │  + ←─────────────────────┘               │
    │    ↓                                     │
    │  ──────────────────────┐                 │
    │    ↓                   │ (residual)      │
    │  LayerNorm             │                 │
    │    ↓                   │                 │
    │  Feed-Forward          │                 │
    │    ↓                   │                 │
    │  + ←───────────────────┘                 │
    │    ↓                                     │
    │  Output                                  │
    └─────────────────────────────────────────┘
    """

    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()

        # Layer Normalization — keeps numbers stable
        # Like a thermostat keeping temperature steady
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # Self-Attention — "gather information from others"
        self.attention = MultiHeadAttention(n_embd, n_heads, block_size)

        # Feed-Forward — "think about what you gathered"
        self.feed_forward = FeedForward(n_embd)

    def forward(self, x):
        """
        Process input through attention and feed-forward,
        with residual connections.

        The residual connection (x + ...) is CRUCIAL:
        - It lets gradients flow easily during training
        - It means the block only needs to learn the DIFFERENCE
        - Like: "keep what you know, and ADD new insights"
        """

        # Attention with residual connection
        # x + attention(norm(x)) means:
        # "Take original x, AND ADD what attention found"
        x = x + self.attention(self.ln1(x))

        # Feed-forward with residual connection
        # Same idea: "Take what you have, AND ADD what FF computed"
        x = x + self.feed_forward(self.ln2(x))

        return x


class BabyLLM(nn.Module):
    """
    🍼 THE COMPLETE BABY LLM!

    This is our full language model that combines everything:
    1. Embeddings (Step 3)
    2. Multiple Transformer Blocks (Step 4 + 5)
    3. Output layer (predicts next character)

    It takes in character IDs and outputs PROBABILITIES
    for what the next character should be!
    """

    def __init__(self, vocab_size, n_embd=64, n_heads=4, n_layers=4, block_size=32):
        """
        Args:
            vocab_size: Number of unique characters (~30)
            n_embd:     Embedding dimension (64)
            n_heads:    Number of attention heads per block (4)
            n_layers:   Number of transformer blocks (4)
            block_size: Maximum sequence length (32)

        For comparison:
            GPT-3:  vocab=50257, n_embd=12288, n_heads=96, n_layers=96, block_size=2048
            Ours:   vocab=~30,   n_embd=64,    n_heads=4,  n_layers=4,  block_size=32
        """
        super().__init__()

        self.block_size = block_size

        # STEP 1: Embeddings — turn character IDs into rich vectors
        self.embeddings = BabyEmbedding(vocab_size, n_embd, block_size)

        # STEP 2: Stack of Transformer Blocks — the "brain"
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_heads, block_size)
            for _ in range(n_layers)
        ])

        # STEP 3: Final layer norm
        self.ln_final = nn.LayerNorm(n_embd)

        # STEP 4: Output layer — converts embeddings to character probabilities
        # Maps from embedding space (64) to vocabulary space (30)
        # Output: for each position, a score for EACH possible next character
        self.output_head = nn.Linear(n_embd, vocab_size)

        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """
        The forward pass — from character IDs to predictions!

        Args:
            idx:     Input character IDs, shape (B, T)
            targets: Expected next characters, shape (B, T) — used for training

        Returns:
            logits: Raw prediction scores, shape (B, T, vocab_size)
            loss:   How wrong our predictions are (only if targets given)

        ┌──────────────────────────────────────────────┐
        │  Input: [2, 7, 4, 1]  (character IDs)        │
        │      ↓                                        │
        │  Embeddings: (B, T, 64)                       │
        │      ↓                                        │
        │  Transformer Block 1: (B, T, 64)              │
        │      ↓                                        │
        │  Transformer Block 2: (B, T, 64)              │
        │      ↓                                        │
        │  Transformer Block 3: (B, T, 64)              │
        │      ↓                                        │
        │  Transformer Block 4: (B, T, 64)              │
        │      ↓                                        │
        │  Layer Norm: (B, T, 64)                       │
        │      ↓                                        │
        │  Output Head: (B, T, vocab_size)              │
        │      ↓                                        │
        │  Logits: scores for each possible next char!  │
        └──────────────────────────────────────────────┘
        """

        # Get embeddings (token + position)
        x = self.embeddings(idx)          # (B, T, n_embd)

        # Pass through all transformer blocks
        x = self.blocks(x)               # (B, T, n_embd)

        # Final layer norm
        x = self.ln_final(x)             # (B, T, n_embd)

        # Project to vocabulary size — get prediction scores
        logits = self.output_head(x)     # (B, T, vocab_size)

        # Calculate loss if we have targets
        loss = None
        if targets is not None:
            # Reshape for cross-entropy loss
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            # Cross-entropy loss measures how wrong our predictions are
            # Lower loss = better predictions!
            # It's like a teacher grading your answers:
            #   - Confident and correct → low loss ✅
            #   - Confident and wrong → high loss ❌
            #   - Unsure → medium loss 🤷
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def count_parameters(self):
        """Count and display the number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        return total


def demo_transformer():
    """Show the complete model architecture."""

    print("=" * 60)
    print("🧠 STEP 5: The Complete Baby LLM Architecture")
    print("=" * 60)

    vocab_size = 30
    model = BabyLLM(vocab_size, n_embd=64, n_heads=4, n_layers=4, block_size=32)

    print(f"\n📐 Model Architecture:")
    print(f"   Vocabulary size:    {vocab_size} characters")
    print(f"   Embedding dimension: 64")
    print(f"   Attention heads:     4")
    print(f"   Transformer blocks:  4")
    print(f"   Max sequence length: 32")
    print(f"\n📊 Total parameters:   {model.count_parameters():,}")
    print(f"   (GPT-3 has 175,000,000,000 — that's 175 BILLION!)")

    # Test with fake input
    fake_input = torch.randint(0, vocab_size, (2, 16))  # 2 sequences, 16 chars each
    fake_targets = torch.randint(0, vocab_size, (2, 16))

    logits, loss = model(fake_input, fake_targets)

    print(f"\n🧪 Test run:")
    print(f"   Input shape:   {fake_input.shape}  (2 sequences × 16 characters)")
    print(f"   Output shape:  {logits.shape}  (2 × 16 × {vocab_size} predictions)")
    print(f"   Initial loss:  {loss.item():.4f}")
    print(f"   (Random guess loss would be: {-1 * torch.log(torch.tensor(1.0/vocab_size)).item():.4f})")

    print(f"\n📋 Model layers:")
    for name, param in model.named_parameters():
        print(f"   {name:45s} → shape {list(param.shape)}")

    print("\n" + "=" * 60)
    print("✅ Baby LLM architecture is complete! Ready for training!")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════
# 🏃 RUN THIS FILE to see the model architecture
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo_transformer()
