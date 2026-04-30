"""
╔══════════════════════════════════════════════════════════════╗
║           STEP 4: SELF-ATTENTION  👀                         ║
║           "The Secret Sauce of Transformers"                 ║
╚══════════════════════════════════════════════════════════════╝

🎯 WHAT IS SELF-ATTENTION?
===========================
Self-Attention lets each character LOOK AT other characters
and decide which ones are IMPORTANT for predicting the next one.

🏫 CLASSROOM ANALOGY:
  Imagine you're in a classroom and the teacher asks a question.
  You look around at your classmates:
    - "Hmm, Alice knows math, she might help..."     (high attention)
    - "Bob is sleeping, probably not useful..."        (low attention)
    - "Carol answered a similar question before!"      (high attention)

  Self-Attention does the SAME thing with characters!
  Each character "looks at" all previous characters and decides
  which ones are most relevant.

🔑 THE KEY IDEA: Query, Key, Value (Q, K, V)
=============================================
  Think of it like a LIBRARY:

  📚 Query (Q) = "What am I looking for?"
     → You walk into the library with a question

  🏷️  Key (K) = "What does each book contain?"
     → Each book has a label describing its content

  📖 Value (V) = "What's actually in the book?"
     → The actual information inside the book

  Process:
  1. Compare your QUERY with each book's KEY
  2. Books with matching keys get HIGH attention scores
  3. Read the VALUES of the high-scoring books
  4. Combine the information!

🎭 WHY "MULTI-HEAD" ATTENTION?
================================
  Instead of one person searching the library,
  send MULTIPLE people (heads), each looking for different things!

  - Head 1 might focus on: "What noun comes before me?"
  - Head 2 might focus on: "What verb is nearby?"
  - Head 3 might focus on: "Is there a pattern repeating?"

  Then combine all their findings!

⚠️ THE MASK (Causal / Autoregressive):
=======================================
  IMPORTANT RULE: When predicting the NEXT character,
  you can only look at PREVIOUS characters, not future ones!

  "the cat" → when predicting what comes after 'c',
  you can see 't', 'h', 'e', ' ', 'c' but NOT 'a' or 't'

  We enforce this with a MASK (like covering future answers on a test!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleAttentionHead(nn.Module):
    """
    ONE attention head — like ONE person searching the library.

    This is the fundamental building block of the Transformer!
    """

    def __init__(self, n_embd, head_size, block_size):
        """
        Args:
            n_embd:     Size of input embeddings (64)
            head_size:  Size of this head's output (n_embd // n_heads)
            block_size: Maximum sequence length (32)
        """
        super().__init__()

        # These are the Q, K, V projections
        # They're just matrix multiplications (linear layers)
        # Think of them as "lenses" that look at the data differently

        self.query = nn.Linear(n_embd, head_size, bias=False)  # "What am I looking for?"
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # "What do I contain?"
        self.value = nn.Linear(n_embd, head_size, bias=False)  # "What's my actual info?"

        # The MASK — prevents looking at future characters
        # register_buffer means: "save this, but don't train it"
        # tril = lower triangular matrix (1s below diagonal, 0s above)
        #
        # For block_size=4, the mask looks like:
        # [[1, 0, 0, 0],    ← position 0 can only see position 0
        #  [1, 1, 0, 0],    ← position 1 can see positions 0,1
        #  [1, 1, 1, 0],    ← position 2 can see positions 0,1,2
        #  [1, 1, 1, 1]]    ← position 3 can see all positions
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        """
        Perform self-attention!

        Args:
            x: Input tensor of shape (batch, sequence_length, n_embd)

        Returns:
            Output tensor of shape (batch, sequence_length, head_size)

        The full process:
        ┌──────────────────────────────────────────────┐
        │  1. Create Q, K, V from input                │
        │  2. Compute attention scores: Q × K^T        │
        │  3. Scale the scores (for stability)         │
        │  4. Apply mask (hide future)                 │
        │  5. Softmax (turn scores into probabilities) │
        │  6. Multiply by V (get weighted values)      │
        └──────────────────────────────────────────────┘
        """
        B, T, C = x.shape  # Batch, Time (seq length), Channels (embedding)

        # Step 1: Create Q, K, V
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Step 2: Compute attention scores
        # "How much should each position attend to each other position?"
        # Matrix multiply Q with K-transposed
        # Result: (B, T, T) — a T×T attention matrix!
        scores = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T) → (B, T, T)

        # Step 3: Scale by sqrt(head_size)
        # WHY? Without scaling, the dot products can get very large,
        # which makes softmax produce very peaked distributions.
        # Dividing by sqrt(d) keeps the values in a nice range.
        # This is the "Scaled" in "Scaled Dot-Product Attention"
        head_size = k.shape[-1]
        scores = scores / math.sqrt(head_size)

        # Step 4: Apply the causal mask
        # Where mask is 0, set score to -infinity
        # After softmax, -infinity becomes 0 (no attention to future!)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # Step 5: Softmax — turn scores into probabilities (0 to 1, sum to 1)
        # Now each row tells us: "how much attention to pay to each position"
        attention_weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # Step 6: Weighted sum of values
        # Multiply attention weights by values to get the output
        # High attention = take more of that position's value
        out = attention_weights @ v  # (B, T, T) @ (B, T, head_size) → (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """
    MULTIPLE attention heads working in parallel!

    Like sending 4 researchers to the library,
    each looking for different information,
    then combining their findings.
    """

    def __init__(self, n_embd, n_heads, block_size):
        """
        Args:
            n_embd:     Total embedding size (64)
            n_heads:    Number of attention heads (4)
            block_size: Maximum sequence length (32)
        """
        super().__init__()

        # Each head gets a portion of the embedding dimension
        # If n_embd=64 and n_heads=4, each head works with 16 dimensions
        head_size = n_embd // n_heads

        # Create multiple heads
        self.heads = nn.ModuleList([
            SingleAttentionHead(n_embd, head_size, block_size)
            for _ in range(n_heads)
        ])

        # After concatenating all heads, project back to n_embd
        # This combines the findings from all heads
        self.projection = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        """
        Run all attention heads and combine results.

        ┌─────────────────────────────────────────┐
        │  Input: (B, T, 64)                      │
        │                                          │
        │  Head 1: (B, T, 16) ─┐                  │
        │  Head 2: (B, T, 16) ─┤ Concatenate      │
        │  Head 3: (B, T, 16) ─┤ → (B, T, 64)    │
        │  Head 4: (B, T, 16) ─┘                  │
        │                                          │
        │  Project: (B, T, 64) → (B, T, 64)       │
        └─────────────────────────────────────────┘
        """
        # Run each head on the same input
        head_outputs = [head(x) for head in self.heads]

        # Concatenate along the last dimension
        concatenated = torch.cat(head_outputs, dim=-1)  # (B, T, n_embd)

        # Final projection
        out = self.projection(concatenated)  # (B, T, n_embd)

        return out


def demo_attention():
    """Visualize how attention works."""

    print("=" * 60)
    print("👀 STEP 4: Understanding Self-Attention")
    print("=" * 60)

    n_embd = 32
    n_heads = 4
    block_size = 8

    # Create a multi-head attention layer
    mha = MultiHeadAttention(n_embd, n_heads, block_size)

    # Fake input: 1 sequence of 6 characters, each with 32-dim embedding
    fake_input = torch.randn(1, 6, n_embd)

    print(f"\n📥 Input shape:  {fake_input.shape}")
    print(f"   (1 sequence, 6 characters, {n_embd}-dim embeddings)")

    output = mha(fake_input)

    print(f"\n📤 Output shape: {output.shape}")
    print(f"   (same shape! but now each character 'knows about' others)")

    print(f"\n🔍 What happened inside:")
    print(f"   • {n_heads} attention heads ran in parallel")
    print(f"   • Each head had size {n_embd // n_heads}")
    print(f"   • Each character looked at previous characters")
    print(f"   • Results were combined")

    # Show the causal mask
    print(f"\n🎭 The Causal Mask (for 6 positions):")
    mask = torch.tril(torch.ones(6, 6))
    for i in range(6):
        row = ""
        for j in range(6):
            row += " ✅" if mask[i][j] == 1 else " ❌"
            # ✅ = can see, ❌ = hidden (future)
        print(f"   Position {i}: {row}")
    print(f"   (✅ = can attend, ❌ = masked/hidden)")

    print("\n" + "=" * 60)
    print("✅ Self-Attention ready! Characters can now 'talk' to each other.")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════
# 🏃 RUN THIS FILE to see attention in action
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo_attention()
