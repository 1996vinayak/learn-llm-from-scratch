"""
╔══════════════════════════════════════════════════════════════╗
║     ADVANCED MODEL — Dropout, GELU, KV-Cache & More!        ║
╚══════════════════════════════════════════════════════════════╝

What's new compared to Baby LLM:

1. DROPOUT — Randomly "turns off" neurons during training
   → Prevents memorization (overfitting)
   → Like studying with random pages missing — forces deeper understanding!

2. GELU activation (instead of ReLU)
   → Smoother than ReLU, used by GPT-2/3/4
   → ReLU: hard cutoff at 0
   → GELU: smooth curve, allows small negative values

3. KV-CACHE — Caches Key and Value computations
   → During generation, we recompute K,V for ALL previous tokens each step
   → KV-Cache saves them so we only compute the NEW token's K,V
   → Makes generation ~10x faster!

4. GRADIENT CLIPPING — Prevents "exploding gradients"
   → Sometimes gradients get HUGE and destabilize training
   → We clip them to a maximum value (like a speed limit!)

5. WEIGHT INITIALIZATION — Better starting weights
   → Random isn't always best
   → Xavier/Kaiming initialization helps training converge faster
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImprovedAttentionHead(nn.Module):
    """
    Attention head with DROPOUT and optional KV-CACHE.

    New concepts:
    - Dropout on attention weights (randomly zero out some attention)
    - KV-Cache for faster generation
    """

    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

        # NEW: Dropout — randomly zeros out attention weights during training
        # This prevents the model from relying too heavily on any single position
        self.attn_dropout = nn.Dropout(dropout)
        self.head_size = head_size

        # KV-Cache storage (used during generation)
        self._cache_k = None
        self._cache_v = None

    def forward(self, x, use_cache=False):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # KV-CACHE: During generation, append to cached K,V
        if use_cache and self._cache_k is not None:
            # Only compute K,V for the NEW token (last position)
            # Append to the cache of all previous tokens
            k = torch.cat([self._cache_k, k], dim=1)
            v = torch.cat([self._cache_v, v], dim=1)

        if use_cache:
            self._cache_k = k.detach()
            self._cache_v = v.detach()

        T_kv = k.shape[1]

        # Attention scores
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)

        # Causal mask (adjusted for cache size)
        if T_kv > T:
            # During cached generation, Q is size 1, K/V is full history
            # No mask needed for single query token
            pass
        else:
            scores = scores.masked_fill(self.mask[:T, :T_kv] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        # NEW: Apply dropout to attention weights
        weights = self.attn_dropout(weights)

        out = weights @ v
        return out

    def clear_cache(self):
        """Clear the KV cache (call before new generation)."""
        self._cache_k = None
        self._cache_v = None


class ImprovedMultiHeadAttention(nn.Module):
    """Multi-head attention with dropout."""

    def __init__(self, n_embd, n_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([
            ImprovedAttentionHead(n_embd, head_size, block_size, dropout)
            for _ in range(n_heads)
        ])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)  # Dropout on output too

    def forward(self, x, use_cache=False):
        head_outputs = [head(x, use_cache) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.projection(concatenated))
        return out

    def clear_cache(self):
        for head in self.heads:
            head.clear_cache()


class ImprovedFeedForward(nn.Module):
    """
    Feed-forward with GELU activation and dropout.

    GELU vs ReLU:
      ReLU(x) = max(0, x)           — hard cutoff at 0
      GELU(x) = x * Φ(x)           — smooth curve
                                      (Φ = cumulative normal distribution)

    GELU is smoother and allows small negative values through,
    which helps the model learn more nuanced patterns.
    GPT-2, GPT-3, BERT all use GELU!
    """

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # ← NEW: GELU instead of ReLU
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),  # ← NEW: Dropout
        )

    def forward(self, x):
        return self.net(x)


class ImprovedTransformerBlock(nn.Module):
    """Transformer block with all improvements."""

    def __init__(self, n_embd, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attention = ImprovedMultiHeadAttention(n_embd, n_heads, block_size, dropout)
        self.feed_forward = ImprovedFeedForward(n_embd, dropout)

    def forward(self, x, use_cache=False):
        x = x + self.attention(self.ln1(x), use_cache)
        x = x + self.feed_forward(self.ln2(x))
        return x

    def clear_cache(self):
        self.attention.clear_cache()


class TeenageLLM(nn.Module):
    """
    The improved "Teenage" LLM with all advanced features!

    Improvements over Baby LLM:
    - Dropout everywhere (prevents overfitting)
    - GELU activation (smoother, better gradients)
    - KV-Cache support (faster generation)
    - Better weight initialization
    """

    def __init__(self, vocab_size, n_embd=128, n_heads=4, n_layers=6,
                 block_size=64, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        # Embeddings with dropout
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ImprovedTransformerBlock(n_embd, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(n_embd)
        self.output_head = nn.Linear(n_embd, vocab_size)

        # NEW: Better weight initialization
        self.apply(self._init_weights)

        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        """
        Initialize weights using techniques from GPT-2.

        Why? Random initialization can lead to:
        - Very large or very small activations
        - Slow or unstable training

        Xavier/normal initialization keeps values in a good range.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, use_cache=False):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.embed_dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x, use_cache)

        x = self.ln_final(x)
        logits = self.output_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def clear_cache(self):
        for block in self.blocks:
            block.clear_cache()


def demo_model():
    print("=" * 60)
    print("🧠 Teenage LLM — Advanced Model Architecture")
    print("=" * 60)

    model = TeenageLLM(vocab_size=100, n_embd=128, n_heads=4,
                       n_layers=6, block_size=64, dropout=0.1)

    print(f"\n📐 Architecture:")
    print(f"   Embedding dim:  128 (was 64)")
    print(f"   Heads:          4")
    print(f"   Layers:         6 (was 4)")
    print(f"   Block size:     64 (was 32)")
    print(f"   Dropout:        0.1 (10% neurons randomly off)")
    print(f"   Activation:     GELU (was ReLU)")
    print(f"   Parameters:     {model.n_params:,}")

    # Test forward pass
    x = torch.randint(0, 100, (2, 32))
    y = torch.randint(0, 100, (2, 32))
    logits, loss = model(x, y)
    print(f"\n🧪 Test: input {x.shape} → output {logits.shape}, loss={loss.item():.4f}")
    print("✅ Model works!")


if __name__ == "__main__":
    demo_model()
