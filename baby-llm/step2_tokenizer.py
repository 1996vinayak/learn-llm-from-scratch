"""
╔══════════════════════════════════════════════════════════════╗
║           STEP 2: TOKENIZER  🔤                              ║
║           "Cutting Text Into Pieces"                         ║
╚══════════════════════════════════════════════════════════════╝

🎯 WHAT IS A TOKENIZER?
========================
A tokenizer breaks text into small pieces called "tokens".
Then it gives each piece a NUMBER (because computers only understand numbers!)

Think of it like this:
  - You have a sentence: "the cat sat"
  - Tokenizer cuts it into pieces: ["t", "h", "e", " ", "c", "a", "t", ...]
  - Then assigns numbers:           [ 2,   4,   1,   0,   3,   7,   2, ...]

🍕 PIZZA ANALOGY:
  - You can't eat a whole pizza at once
  - You cut it into SLICES (tokens!)
  - Each slice gets a NUMBER (slice 1, slice 2, etc.)
  - Now you can eat (process) it piece by piece!

📊 TYPES OF TOKENIZERS:
========================
  1. CHARACTER-LEVEL  → Each letter is a token     ("h","e","l","l","o")
  2. WORD-LEVEL       → Each word is a token        ("hello", "world")
  3. SUBWORD (BPE)    → Mix of both                 ("hel", "lo", "wor", "ld")

  Real LLMs use SUBWORD tokenizers (like BPE).
  We'll use CHARACTER-LEVEL because it's simplest to understand!

🤔 WHY CHARACTERS?
========================
  - Tiny vocabulary (just ~30 characters vs 50,000+ for real LLMs)
  - Easy to understand
  - Same concept, just smaller scale
"""


class BabyTokenizer:
    """
    Our simple character-level tokenizer.

    It does TWO things:
    1. ENCODE: Turn text → numbers  ("hello" → [7, 4, 11, 11, 14])
    2. DECODE: Turn numbers → text  ([7, 4, 11, 11, 14] → "hello")

    Real tokenizers (like GPT's tiktoken) do the same thing,
    just with subwords instead of characters!
    """

    def __init__(self, text):
        """
        Build the tokenizer from our training text.

        Steps:
        1. Find all unique characters in the text
        2. Give each character a unique number (its "token ID")
        3. Create lookup tables for encode/decode
        """

        # Step 1: Find all unique characters
        # set() removes duplicates, sorted() puts them in order
        self.chars = sorted(set(text))

        # This is our "vocabulary" — all the characters our model knows
        self.vocab_size = len(self.chars)

        # Step 2: Create lookup tables
        # "char to index" — given a character, what's its number?
        # Example: {'a': 0, 'b': 1, 'c': 2, ...}
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}

        # "index to char" — given a number, what's the character?
        # Example: {0: 'a', 1: 'b', 2: 'c', ...}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        """
        ENCODE: Turn text into a list of numbers.

        "cat" → [3, 1, 20]  (example numbers)

        This is like translating English to "Computer Language"!
        """
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """
        DECODE: Turn a list of numbers back into text.

        [3, 1, 20] → "cat"  (example numbers)

        This is like translating "Computer Language" back to English!
        """
        return "".join([self.idx_to_char[i] for i in indices])

    def show_vocabulary(self):
        """Print the vocabulary so we can see it."""

        print("=" * 60)
        print("🔤 STEP 2: Our Tokenizer's Vocabulary")
        print("=" * 60)

        print(f"\n📊 Vocabulary size: {self.vocab_size} characters")
        print(f"\n📋 Character → Number mapping:")

        for ch, idx in self.char_to_idx.items():
            # Make whitespace characters visible
            display_ch = repr(ch) if ch in " \n\t" else f" {ch} "
            print(f"   {display_ch} → {idx}")

    def demo(self):
        """Show how encoding and decoding works."""

        print(f"\n🔄 DEMO: Encoding and Decoding")
        print("-" * 40)

        sample = "the cat"
        encoded = self.encode(sample)
        decoded = self.decode(encoded)

        print(f"   Original text:  '{sample}'")
        print(f"   Encoded (nums): {encoded}")
        print(f"   Decoded (text): '{decoded}'")
        print(f"   Match? {sample == decoded} ✅")

        print("\n   Step by step encoding:")
        for ch in sample:
            display = repr(ch) if ch == " " else f"'{ch}'"
            print(f"   {display:6s} → {self.char_to_idx[ch]}")

        print("\n" + "=" * 60)
        print("✅ Tokenizer is ready! We can convert text ↔ numbers.")
        print("=" * 60)


# ═══════════════════════════════════════════════════════════
# 🏃 RUN THIS FILE to see the tokenizer in action
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    from step1_data import get_training_data

    text = get_training_data()
    tokenizer = BabyTokenizer(text)
    tokenizer.show_vocabulary()
    tokenizer.demo()
