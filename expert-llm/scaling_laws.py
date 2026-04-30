"""
Scaling Laws (Chinchilla)
=========================
Key question: Given a compute budget, how should you
split it between model size and training data?

Chinchilla finding (2022):
  Optimal: tokens = 20 * parameters
  
  A 1B model should train on 20B tokens
  A 7B model should train on 140B tokens
  A 70B model should train on 1.4T tokens

Before Chinchilla, models were too big for their data:
  GPT-3 (175B params) trained on only 300B tokens
  Should have been: 175B * 20 = 3.5T tokens!

Loss scales as a power law:
  L(N) ~ N^(-0.076)  (model size)
  L(D) ~ D^(-0.095)  (data size)
  L(C) ~ C^(-0.050)  (compute)
"""

import math


def chinchilla_optimal(compute_flops):
    """Given compute budget, find optimal model size and tokens."""
    # Chinchilla: N_opt ~ C^0.5, D_opt ~ C^0.5
    # Simplified: tokens = 20 * params
    # C ~ 6 * N * D (approximate FLOPs)
    N = (compute_flops / 120) ** 0.5  # params
    D = 20 * N  # tokens
    return N, D


def predict_loss(params, tokens):
    """Predict training loss using scaling laws."""
    # Simplified Chinchilla scaling law
    A, alpha = 406.4, 0.34
    B, beta = 410.7, 0.28
    E = 1.69
    loss = (A / params**alpha) + (B / tokens**beta) + E
    return loss


def demo_scaling():
    print("=" * 60)
    print("Scaling Laws (Chinchilla)")
    print("=" * 60)
    
    print("\n  Chinchilla Rule: tokens = 20 * parameters\n")
    
    models = [
        ("1B", 1e9), ("7B", 7e9), ("13B", 13e9),
        ("70B", 70e9), ("175B", 175e9),
    ]
    
    for name, params in models:
        tokens = 20 * params
        print(f"  {name:>5s} model -> {tokens/1e9:.0f}B tokens optimal")


if __name__ == "__main__":
    demo_scaling()
