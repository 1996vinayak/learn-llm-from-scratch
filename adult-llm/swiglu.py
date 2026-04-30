"""
SwiGLU Activation — Used by LLaMA, PaLM, and Mistral
=====================================================
Baby LLM:    ReLU    → max(0, x)
Teenage LLM: GELU    → x * Phi(x)
Adult LLM:   SwiGLU  → Swish(xW1) * (xW2)

SwiGLU is a "gated" activation:
  - Two linear projections: gate and value
  - Gate uses Swish activation: x * sigmoid(x)
  - Output = Swish(gate) * value

WHY SwiGLU?
  - Consistently outperforms ReLU and GELU in experiments
  - The "gating" mechanism lets the network learn WHAT to let through
  - Like a bouncer at a club: decides what information passes!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    Standard FFN:  Linear -> GELU -> Linear
    SwiGLU FFN:    Two parallel Linears -> SwiGLU gate -> Linear
    """
    
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        # Hidden size is 2/3 of 4*n_embd to keep param count similar
        hidden = int(2 * (4 * n_embd) / 3)
        
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)  # Gate projection
        self.w_up = nn.Linear(n_embd, hidden, bias=False)    # Value projection
        self.w_down = nn.Linear(hidden, n_embd, bias=False)  # Output projection
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU: swish(gate) * value
        gate = F.silu(self.w_gate(x))  # silu = swish = x * sigmoid(x)
        value = self.w_up(x)
        x = gate * value               # Gating!
        x = self.w_down(x)
        return self.dropout(x)


def demo_swiglu():
    print("=" * 60)
    print("SwiGLU Activation Demo")
    print("=" * 60)
    
    n_embd = 64
    x = torch.randn(1, 8, n_embd)
    
    # Compare parameter counts
    standard = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd))
    swiglu = SwiGLUFeedForward(n_embd)
    
    std_params = sum(p.numel() for p in standard.parameters())
    swi_params = sum(p.numel() for p in swiglu.parameters())
    
    print(f"\n  Standard FFN params: {std_params:,}")
    print(f"  SwiGLU FFN params:   {swi_params:,}")
    print(f"  Output shape: {swiglu(x).shape}")
    
    # Show Swish vs ReLU vs GELU
    test = torch.linspace(-3, 3, 7)
    print(f"\n  x:     {[f'{v:.1f}' for v in test.tolist()]}")
    print(f"  ReLU:  {[f'{max(0,v):.2f}' for v in test.tolist()]}")
    print(f"  GELU:  {[f'{v:.2f}' for v in F.gelu(test).tolist()]}")
    print(f"  Swish: {[f'{v:.2f}' for v in F.silu(test).tolist()]}")


if __name__ == "__main__":
    demo_swiglu()
