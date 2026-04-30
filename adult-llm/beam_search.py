"""
Beam Search — Deterministic Text Generation
============================================
Baby/Teenage LLM: Random sampling (temperature, top-k, top-p)
Adult LLM: Beam Search (find the BEST sequence)

HOW IT WORKS:
  Instead of picking ONE token at a time (greedy),
  keep track of the TOP B sequences (beams).
  
  beam_width=3 means: keep 3 best candidates at each step.
  
  Step 1: "The" -> top 3 next tokens:
    Beam 1: "The little" (score: 0.9)
    Beam 2: "The cat"    (score: 0.7)
    Beam 3: "The big"    (score: 0.5)
  
  Step 2: Expand each beam, keep top 3 overall:
    Beam 1: "The little cat"  (score: 0.85)
    Beam 2: "The little dog"  (score: 0.80)
    Beam 3: "The cat sat"     (score: 0.65)
  
  Final: Return the highest-scoring complete sequence!

PROS: More coherent, deterministic output
CONS: Slower, less creative, can be repetitive
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def beam_search(model, tokenizer, start_text="The ",
                max_new_tokens=50, beam_width=3):
    model.eval()
    
    tokens = tokenizer.encode(start_text)
    
    # Initialize beams: (sequence, cumulative_log_prob)
    beams = [(tokens[:], 0.0)]
    
    for step in range(max_new_tokens):
        all_candidates = []
        
        for seq, score in beams:
            input_ids = torch.tensor([seq[-model.block_size:]], dtype=torch.long)
            logits, _ = model(input_ids)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            
            # Get top beam_width tokens for this beam
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)
            
            for i in range(beam_width):
                new_seq = seq + [top_indices[0, i].item()]
                new_score = score + top_log_probs[0, i].item()
                all_candidates.append((new_seq, new_score))
        
        # Keep only top beam_width candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]
    
    # Return best beam
    best_seq, best_score = beams[0]
    return tokenizer.decode(best_seq), best_score


def demo_beam():
    print("=" * 60)
    print("Beam Search Demo")
    print("=" * 60)
    print("\nBeam search keeps multiple candidates at each step.")
    print("It finds the highest-probability SEQUENCE, not just")
    print("the highest-probability next token at each step.")
    print("\nThis is used for translation, summarization, and")
    print("any task where quality > creativity.")


if __name__ == "__main__":
    demo_beam()
