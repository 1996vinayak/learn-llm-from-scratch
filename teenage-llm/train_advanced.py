"""
Advanced Training Loop

New features over Baby LLM:
  - Learning rate scheduling (warmup + cosine decay)
  - Gradient clipping (prevent exploding gradients)
  - Perplexity tracking
  - Train/validation split
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from bpe_tokenizer import BPETokenizer
from advanced_model import TeenageLLM
from lr_scheduler import CosineScheduler
from evaluate import calculate_perplexity


def get_training_data():
    text = """Once upon a time there was a little cat.
The little cat loved to play in the garden.
Every day the little cat would chase butterflies.
The butterflies would fly high in the sky.
The little cat would jump and jump and jump.
But the butterflies were too high for the little cat.
So the little cat would sit in the garden and watch.
The garden was full of flowers and trees.
The little cat loved the flowers in the garden.
Every day was a happy day for the little cat.
The little cat would sleep under the big tree.
The big tree gave shade to the little cat.
When the sun went down the little cat went home.
Home was warm and safe for the little cat.
The little cat would dream about butterflies.
And every morning the little cat would go to the garden again.
The garden was the favorite place of the little cat.
The little cat was happy in the garden every day.
Once upon a time there was a happy little cat.
The happy little cat lived in a beautiful garden."""
    return text


def train_advanced(n_embd=128, n_heads=4, n_layers=6, block_size=64,
                   batch_size=16, peak_lr=3e-4, n_steps=3000,
                   warmup_steps=300, dropout=0.1, grad_clip=1.0):
    
    print("=" * 60)
    print("Advanced Training - Teenage LLM")
    print("=" * 60)
    
    # Data
    text = get_training_data()
    
    # BPE Tokenizer
    print("\n--- Training BPE Tokenizer ---")
    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=50)
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Train/val split (90/10)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    print(f"\n  Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")
    
    # Model
    print("\n--- Creating Model ---")
    model = TeenageLLM(
        vocab_size=tokenizer.vocab_size, n_embd=n_embd,
        n_heads=n_heads, n_layers=n_layers,
        block_size=block_size, dropout=dropout
    )
    print(f"  Parameters: {model.n_params:,}")
    
    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    scheduler = CosineScheduler(optimizer, warmup_steps, n_steps, peak_lr)
    
    # Training loop
    print(f"\n--- Training for {n_steps} steps ---")
    print(f"  Warmup: {warmup_steps} steps")
    print(f"  Gradient clipping: {grad_clip}")
    print(f"  Dropout: {dropout}")
    print("-" * 60)
    
    for step in range(n_steps):
        # Get batch
        starts = torch.randint(0, len(train_data) - block_size - 1, (batch_size,))
        x = torch.stack([train_data[s:s+block_size] for s in starts])
        y = torch.stack([train_data[s+1:s+block_size+1] for s in starts])
        
        # Forward
        logits, loss = model(x, y)
        
        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update
        optimizer.step()
        lr = scheduler.step()
        
        if step % 300 == 0 or step == n_steps - 1:
            val_ppl, val_loss = calculate_perplexity(model, val_data, block_size)
            print(f"  Step {step:5d} | Loss: {loss.item():.4f} | "
                  f"Val PPL: {val_ppl:.2f} | LR: {lr:.6f}")
    
    print("-" * 60)
    print("Training complete!")
    
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train_advanced()
