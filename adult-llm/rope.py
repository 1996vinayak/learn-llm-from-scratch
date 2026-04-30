"""
RoPE - Rotary Position Embeddings
==================================
Baby LLM: Learned position embeddings (a lookup table)
Adult LLM: RoPE (mathematical rotation)

WHY RoPE?
  - Learned embeddings have a FIXED max length
  - RoPE can extrapolate to LONGER sequences!
  - Used by LLaMA, GPT-NeoX, PaLM-2

HOW IT WORKS:
  Instead of ADDING position info, RoPE ROTATES the
  query and key vectors based on their position.
  
  Think of a clock:
    Position 0 = 12 o'clock
    Position 1 = rotated slightly clockwise
    Position 2 = rotated more
    ...
  
  The ANGLE between two positions encodes their
  RELATIVE distance. This is the key insight!
  
  Nearby tokens have small angles between them.
  Far-apart tokens have large angles.
"""

import torch
import math


def precompute_rope_frequencies(dim, max_seq_len, theta=10000.0):
    """
    Precompute the rotation frequencies for RoPE.
    
    Each pair of dimensions gets a different rotation speed.
    Lower dimensions rotate faster (capture local patterns).
    Higher dimensions rotate slower (capture global patterns).
    """
    # Frequencies for each dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # Position indices
    positions = torch.arange(max_seq_len).float()
    
    # Outer product: (max_seq_len, dim/2)
    angles = torch.outer(positions, freqs)
    
    # Complex representation for rotation
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    
    return cos_angles, sin_angles


def apply_rope(x, cos, sin):
    """
    Apply rotary embeddings to input tensor.
    
    Rotates pairs of dimensions:
      [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    
    This is just 2D rotation applied to each pair!
    """
    # Split into pairs
    x_even = x[..., 0::2]  # dimensions 0, 2, 4, ...
    x_odd = x[..., 1::2]   # dimensions 1, 3, 5, ...
    
    # Rotate each pair
    T = x.shape[-2]
    cos_t = cos[:T].unsqueeze(0)  # (1, T, dim/2)
    sin_t = sin[:T].unsqueeze(0)
    
    rotated_even = x_even * cos_t - x_odd * sin_t
    rotated_odd = x_even * sin_t + x_odd * cos_t
    
    # Interleave back
    result = torch.stack([rotated_even, rotated_odd], dim=-1)
    return result.flatten(-2)


def demo_rope():
    print("=" * 60)
    print("RoPE - Rotary Position Embeddings")
    print("=" * 60)
    
    dim = 8
    max_len = 32
    cos, sin = precompute_rope_frequencies(dim, max_len)
    
    # Create sample Q,K vectors
    q = torch.randn(1, 4, dim)  # batch=1, seq=4, dim=8
    k = torch.randn(1, 4, dim)
    
    q_rotated = apply_rope(q, cos, sin)
    k_rotated = apply_rope(k, cos, sin)
    
    print(f"\n  Input Q shape: {q.shape}")
    print(f"  Rotated Q shape: {q_rotated.shape}")
    print(f"  Rotation angles (first 4 positions, dim 0):")
    for i in range(4):
        angle_deg = math.degrees(math.atan2(sin[i, 0].item(), cos[i, 0].item()))
        print(f"    Position {i}: {angle_deg:.1f} degrees")
    print("\n  Nearby positions have similar angles!")
    print("  Far positions have different angles!")


if __name__ == "__main__":
    demo_rope()
