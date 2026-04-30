"""
Run the complete Teenage LLM pipeline!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_advanced import train_advanced
from generate_advanced import demo_generation


def main():
    print()
    print("=" * 60)
    print("  Teenage LLM - Level 2 Language Model!")
    print("  Building on Baby LLM with advanced techniques")
    print("=" * 60)
    
    # Train
    model, tokenizer = train_advanced(
        n_embd=128, n_heads=4, n_layers=6,
        block_size=64, batch_size=16,
        peak_lr=3e-4, n_steps=3000,
        warmup_steps=300, dropout=0.1
    )
    
    # Generate
    demo_generation(model, tokenizer)
    
    print("\n" + "=" * 60)
    print("  Done! You've trained a Teenage LLM!")
    print("=" * 60)


if __name__ == "__main__":
    main()
