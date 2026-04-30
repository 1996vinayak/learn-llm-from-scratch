"""Run all Expert LLM demos."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moe import demo_moe
from lora import demo_lora
from quantization import demo_quantization
from scaling_laws import demo_scaling
from speculative_decoding import demo_speculative
from dpo import demo_dpo

def main():
    print("\n" + "=" * 60)
    print("  Expert LLM - Level 4: Frontier Techniques!")
    print("=" * 60 + "\n")
    
    demo_moe()
    print()
    demo_lora()
    print()
    demo_quantization()
    print()
    demo_scaling()
    print()
    demo_speculative()
    print()
    demo_dpo()
    
    print("\n" + "=" * 60)
    print("  Level 4 Complete! You're an LLM Expert!")
    print("=" * 60)

if __name__ == "__main__":
    main()
