"""
Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
=============================================================
Standard Multi-Head Attention: Each head has its own Q, K, V
  - 4 heads = 4 Q matrices, 4 K matrices, 4 V matrices
  - Total: 12 matrices

Multi-Query Attention (MQA): All heads SHARE K and V
  - 4 heads = 4 Q matrices, 1 K matrix, 1 V matrix
  - Total: 6 matrices (50% less!)
  - Used by: PaLM, Falcon, StarCoder

Grouped-Query Attention (GQA): Groups of heads share K,V
  - 4 heads, 2 KV groups = 4 Q, 2 K, 2 V matrices
  - Total: 8 matrices (middle ground)
  - Used by: LLaMA-2 70B, Mistral

WHY?
  - KV-Cache is the memory bottleneck during generation
  - Fewer K,V heads = smaller cache = more tokens in memory
  - Quality barely drops!
"""

import torch
import torch.nn as nn
import math


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    Set n_kv_heads=1 for MQA, n_kv_heads=n_heads for standard MHA.
    """
    
    def __init__(self, n_embd, n_heads, n_kv_heads, block_size, dropout=0.1):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # how many Q heads per KV head
        self.head_size = n_embd // n_heads
        
        # Q has full heads, K and V have fewer
        self.q_proj = nn.Linear(n_embd, n_heads * self.head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_heads * self.head_size, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
    
    def _repeat_kv(self, x):
        """Repeat KV heads to match Q heads."""
        if self.n_rep == 1:
            return x
        B, T, n_kv, hs = x.shape
        return x.unsqueeze(3).expand(B, T, n_kv, self.n_rep, hs).reshape(B, T, self.n_heads, hs)
    
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_size)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_size)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_size)
        
        # Repeat K,V to match number of Q heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        # Transpose for attention: (B, n_heads, T, head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        
        out = (weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


def demo_gqa():
    print("=" * 60)
    print("Grouped-Query Attention Demo")
    print("=" * 60)
    
    n_embd, n_heads, block_size = 64, 4, 32
    x = torch.randn(1, 8, n_embd)
    
    for n_kv in [4, 2, 1]:
        name = {4: "Standard MHA", 2: "GQA (2 groups)", 1: "MQA (1 group)"}[n_kv]
        gqa = GroupedQueryAttention(n_embd, n_heads, n_kv, block_size)
        params = sum(p.numel() for p in gqa.parameters())
        out = gqa(x)
        print(f"\n  {name}: {params:,} params, output={out.shape}")


if __name__ == "__main__":
    demo_gqa()
