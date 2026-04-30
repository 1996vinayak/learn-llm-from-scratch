"""
Adult LLM Model — Combining All Advanced Techniques
====================================================
This model combines:
  - RoPE (Rotary Position Embeddings)
  - Grouped-Query Attention (GQA)
  - SwiGLU activation
  - RMSNorm (instead of LayerNorm)
  - Dropout and gradient clipping ready
  
This is essentially the LLaMA architecture!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rope import precompute_rope_frequencies, apply_rope
from swiglu import SwiGLUFeedForward


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Used by LLaMA instead of standard LayerNorm.
    
    Simpler and slightly faster than LayerNorm:
    - LayerNorm: normalize, then scale and shift
    - RMSNorm: normalize by RMS only, then scale (no shift)
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class AdultAttention(nn.Module):
    """Attention with RoPE and GQA."""
    
    def __init__(self, n_embd, n_heads, n_kv_heads, block_size, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_size = n_embd // n_heads
        
        self.q_proj = nn.Linear(n_embd, n_heads * self.head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_heads * self.head_size, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        
        # RoPE frequencies
        cos, sin = precompute_rope_frequencies(self.head_size, block_size)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
    
    def _repeat_kv(self, x):
        if self.n_rep == 1:
            return x
        B, n_kv, T, hs = x.shape
        return x.unsqueeze(2).expand(B, n_kv, self.n_rep, T, hs).reshape(B, self.n_heads, T, hs)
    
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_size)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_size)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_size)
        
        # Apply RoPE to Q and K
        q = apply_rope(q.transpose(1, 2), self.rope_cos, self.rope_sin).transpose(1, 2)
        k = apply_rope(k.transpose(1, 2), self.rope_cos, self.rope_sin).transpose(1, 2)
        
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = self._repeat_kv(k.view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2))
        v = self._repeat_kv(v.view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2))
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        
        out = (weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class AdultTransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads, n_kv_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        self.attention = AdultAttention(n_embd, n_heads, n_kv_heads, block_size, dropout)
        self.feed_forward = SwiGLUFeedForward(n_embd, dropout)
    
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class AdultLLM(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_heads=8, n_kv_heads=2,
                 n_layers=8, block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.embed_dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            AdultTransformerBlock(n_embd, n_heads, n_kv_heads, block_size, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_final = RMSNorm(n_embd)
        self.output_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying: share embedding and output weights
        self.output_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed_dropout(self.token_embedding(idx))
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.output_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def demo_model():
    print("=" * 60)
    print("Adult LLM - LLaMA-style Architecture")
    print("=" * 60)
    model = AdultLLM(vocab_size=100)
    print(f"\n  Parameters: {model.n_params:,}")
    print(f"  Architecture: RoPE + GQA + SwiGLU + RMSNorm")
    x = torch.randint(0, 100, (2, 32))
    logits, loss = model(x, x)
    print(f"  Test: {x.shape} -> {logits.shape}, loss={loss.item():.4f}")


if __name__ == "__main__":
    demo_model()
