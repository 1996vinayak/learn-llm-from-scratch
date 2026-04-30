"""
Advanced Text Generation - Top-p Sampling and KV-Cache

NEW CONCEPTS:
  Top-p (Nucleus) Sampling:
    Instead of top-k (pick from K most likely),
    top-p picks from the SMALLEST set of tokens
    whose cumulative probability exceeds p.
    
    Example with p=0.9:
      Token probs: [0.5, 0.25, 0.1, 0.08, 0.04, 0.03]
      Cumulative:  [0.5, 0.75, 0.85, 0.93, ...]
      Top-p keeps: [0.5, 0.25, 0.1, 0.08] (sum > 0.9)
      
    This adapts to the distribution!
    - Confident prediction: only 1-2 tokens kept
    - Uncertain prediction: many tokens kept
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_advanced(model, tokenizer, start_text="The ",
                      max_new_chars=200, temperature=0.8,
                      top_k=None, top_p=None, use_cache=False):
    """
    Advanced generation with top-p sampling and KV-cache.
    
    Args:
        top_p: Nucleus sampling threshold (e.g., 0.9)
        use_cache: Whether to use KV-cache for speed
    """
    model.eval()
    model.clear_cache()
    
    tokens = tokenizer.encode(start_text)
    tokens = torch.tensor([tokens], dtype=torch.long)
    
    for i in range(max_new_chars):
        if use_cache and i > 0:
            context = tokens[:, -1:]  # Only last token with cache
        else:
            context = tokens[:, -model.block_size:]
        
        logits, _ = model(context, use_cache=use_cache)
        logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        if top_k is not None:
            top_values, _ = torch.topk(logits, top_k)
            logits[logits < top_values[:, [-1]]] = float("-inf")
        
        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative prob above threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            
            # Scatter back to original ordering
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    
    model.clear_cache()
    return tokenizer.decode(tokens[0].tolist())


def demo_generation(model, tokenizer):
    print("\n" + "=" * 60)
    print("Advanced Text Generation Demo")
    print("=" * 60)
    
    print("\n--- Top-p=0.9, temp=0.8 ---")
    text = generate_advanced(model, tokenizer, "The little ",
                            temperature=0.8, top_p=0.9, max_new_chars=150)
    print(text)
    
    print("\n--- Top-k=5, temp=0.5 ---")
    text = generate_advanced(model, tokenizer, "The little ",
                            temperature=0.5, top_k=5, max_new_chars=150)
    print(text)
    
    print("\n--- With KV-Cache (faster!) ---")
    text = generate_advanced(model, tokenizer, "Once upon ",
                            temperature=0.8, top_p=0.9,
                            use_cache=True, max_new_chars=150)
    print(text)
