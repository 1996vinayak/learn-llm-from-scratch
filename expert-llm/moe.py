"""
Mixture of Experts (MoE)
========================
Standard Transformer: Every token goes through ALL parameters.
MoE Transformer: Each token only uses a FEW "expert" networks.

How it works:
  - Replace the single Feed-Forward network with N experts
  - A "router" decides which experts each token uses
  - Typically top-2 out of 8 experts per token

Why? You get a HUGE model (many params) but only activate
a SMALL part for each token (fast inference).

GPT-4 is rumored to be 8x220B MoE = 1.76T total params
but only ~220B active per token!

Mixtral 8x7B: 8 experts of 7B each = 46.7B total
but only 12.9B active per token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """One expert = one Feed-Forward network."""
    def __init__(self, n_embd, hidden_mult=4):
        super().__init__()
        hidden = n_embd * hidden_mult
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with top-k routing.
    """
    def __init__(self, n_embd, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # The router: decides which experts to use
        self.router = nn.Linear(n_embd, n_experts, bias=False)
        
        # The experts
        self.experts = nn.ModuleList([
            Expert(n_embd) for _ in range(n_experts)
        ])
    
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        
        # Router scores
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs (simplified)
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_probs[:, i].unsqueeze(-1)
            for e_idx in range(self.n_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += expert_weight[mask] * expert_output
        
        return output.view(B, T, C)


def demo_moe():
    print("=" * 60)
    print("Mixture of Experts Demo")
    print("=" * 60)
    
    n_embd = 64
    moe = MoELayer(n_embd, n_experts=8, top_k=2)
    x = torch.randn(1, 8, n_embd)
    out = moe(x)
    
    total_params = sum(p.numel() for p in moe.parameters())
    active_params = sum(p.numel() for p in moe.experts[0].parameters()) * 2
    
    print(f"\n  Total params: {total_params:,}")
    print(f"  Active per token: ~{active_params:,} (top-2 of 8 experts)")
    print(f"  Efficiency: {active_params/total_params*100:.1f}% active")
    print(f"  Output: {out.shape}")


if __name__ == "__main__":
    demo_moe()
