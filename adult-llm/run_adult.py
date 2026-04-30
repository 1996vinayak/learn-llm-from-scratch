"""Run the Adult LLM demo."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rope import demo_rope
from multi_query_attention import demo_gqa
from swiglu import demo_swiglu
from beam_search import demo_beam
from adult_model import demo_model

def main():
    print("\n" + "=" * 60)
    print("  Adult LLM - Level 3: Production Techniques!")
    print("=" * 60 + "\n")
    
    demo_rope()
    print()
    demo_gqa()
    print()
    demo_swiglu()
    print()
    demo_beam()
    print()
    demo_model()
    
    print("\n" + "=" * 60)
    print("  Level 3 Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
