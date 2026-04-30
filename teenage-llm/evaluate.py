"""
Model Evaluation - Perplexity

PERPLEXITY: How "surprised" the model is by the text.
  - Low perplexity = model predicts well (not surprised)
  - High perplexity = model is confused (very surprised)
  
  Perplexity = exp(average cross-entropy loss)
  
  Intuition: If perplexity = 10, the model is as confused
  as if it had to choose between 10 equally likely options
  at each step.
  
  Good perplexity for our tiny model: < 5
  GPT-3 perplexity on test data: ~20-30
"""

import torch
import math


@torch.no_grad()
def calculate_perplexity(model, data, block_size, batch_size=8):
    """
    Calculate perplexity on a dataset.
    
    Perplexity = exp(average_loss)
    Lower is better!
    """
    model.eval()
    total_loss = 0
    n_batches = 0
    
    for i in range(0, len(data) - block_size - 1, block_size * batch_size):
        batch_x = []
        batch_y = []
        for j in range(batch_size):
            start = i + j * block_size
            if start + block_size + 1 > len(data):
                break
            batch_x.append(data[start:start + block_size])
            batch_y.append(data[start + 1:start + block_size + 1])
        
        if not batch_x:
            break
            
        x = torch.stack(batch_x)
        y = torch.stack(batch_y)
        
        _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(avg_loss)
    
    model.train()
    return perplexity, avg_loss


def demo_perplexity():
    print("=" * 60)
    print("Perplexity - Measuring Model Quality")
    print("=" * 60)
    print("\nPerplexity = exp(average loss)")
    print("Lower = better!")
    print("\nExamples:")
    for loss in [3.5, 2.0, 1.0, 0.5, 0.2]:
        ppl = math.exp(loss)
        print(f"  Loss={loss:.1f} -> Perplexity={ppl:.1f}")
    print("\nA perplexity of 10 means the model is as confused")
    print("as choosing between 10 equally likely options.")


if __name__ == "__main__":
    demo_perplexity()
