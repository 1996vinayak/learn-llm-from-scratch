"""
LoRA - Low-Rank Adaptation
===========================
Problem: Fine-tuning a 7B model requires updating ALL 7B parameters.
That needs ~28GB of GPU memory just for the weights!

LoRA solution: Instead of updating the full weight matrix W,
add a small low-rank update: W + A*B

Original: W is (d, d) = d*d parameters
LoRA: A is (d, r) and B is (r, d) where r << d
      Total new params: 2*d*r (much smaller!)

Example: d=4096, r=16
  Full fine-tune: 4096*4096 = 16.7M params per layer
  LoRA: 2*4096*16 = 131K params per layer (128x less!)

The original weights are FROZEN. Only A and B are trained.
This means you can fine-tune on a laptop!
"""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """A linear layer with LoRA adaptation."""
    
    def __init__(self, original_linear, rank=16, alpha=32):
        super().__init__()
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # Freeze original weights
        self.original = original_linear
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * (1/math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Original output + LoRA update
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out


def demo_lora():
    print("=" * 60)
    print("LoRA - Low-Rank Adaptation Demo")
    print("=" * 60)
    
    d = 4096
    original = nn.Linear(d, d)
    lora = LoRALinear(original, rank=16)
    
    orig_params = sum(p.numel() for p in original.parameters())
    lora_params = lora.lora_A.numel() + lora.lora_B.numel()
    
    print(f"\n  Original params: {orig_params:,}")
    print(f"  LoRA params:     {lora_params:,}")
    print(f"  Reduction:       {orig_params/lora_params:.0f}x fewer trainable params!")
    print(f"  LoRA rank:       16")
    
    x = torch.randn(1, 8, d)
    out = lora(x)
    print(f"  Output shape:    {out.shape}")


if __name__ == "__main__":
    demo_lora()
