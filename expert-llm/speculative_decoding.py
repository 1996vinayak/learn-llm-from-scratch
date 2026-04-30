"""
Speculative Decoding — 2-3x Faster Generation
==============================================
Problem: LLM generation is slow because it's sequential
(one token at a time, each needs a full forward pass).

Idea: Use a SMALL fast model to "draft" several tokens,
then verify them all at once with the BIG model.

Steps:
  1. Small model generates K draft tokens quickly
  2. Big model verifies all K tokens in ONE forward pass
  3. Accept tokens that match, reject the rest
  4. Continue from the last accepted token

Why it works:
  - Small model is often correct (easy tokens)
  - Big model can verify K tokens as fast as generating 1
  - Net result: 2-3x speedup!
"""


def demo_speculative():
    print("=" * 60)
    print("Speculative Decoding Demo")
    print("=" * 60)
    
    print("""
  Without speculative decoding:
    Step 1: Big model generates "The"     (100ms)
    Step 2: Big model generates "little"  (100ms)
    Step 3: Big model generates "cat"     (100ms)
    Total: 300ms for 3 tokens

  With speculative decoding (K=3):
    Step 1: Small model drafts "The little cat" (10ms)
    Step 2: Big model verifies all 3 at once    (100ms)
    Step 3: All 3 accepted!
    Total: 110ms for 3 tokens (2.7x faster!)

  If draft is wrong:
    Step 1: Small model drafts "The big dog"    (10ms)
    Step 2: Big model verifies: "The" OK, "big" WRONG
    Step 3: Accept "The", regenerate from there
    Still faster than generating each one!
    """)


if __name__ == "__main__":
    demo_speculative()
