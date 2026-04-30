"""
╔══════════════════════════════════════════════════════════════╗
║     BPE TOKENIZER — How GPT Actually Tokenizes Text!        ║
╚══════════════════════════════════════════════════════════════╝

In Baby LLM, we used character-level tokenization:
  "hello" → ['h','e','l','l','o'] → [7, 4, 11, 11, 14]

Real LLMs use BPE (Byte-Pair Encoding):
  "hello" → ['hel', 'lo'] → [1542, 328]

WHY BPE?
  - Characters are too small (model needs many steps for one word)
  - Words are too big (vocabulary would be enormous)
  - BPE finds the sweet spot: common SUBWORDS

HOW BPE WORKS:
  1. Start with individual characters
  2. Find the most frequent PAIR of adjacent tokens
  3. MERGE that pair into a new token
  4. Repeat until you reach desired vocabulary size

  Example:
    Text: "low lower lowest"
    Step 0: ['l','o','w',' ','l','o','w','e','r',' ','l','o','w','e','s','t']
    Most frequent pair: ('l','o') → merge into 'lo'
    Step 1: ['lo','w',' ','lo','w','e','r',' ','lo','w','e','s','t']
    Most frequent pair: ('lo','w') → merge into 'low'
    Step 2: ['low',' ','low','e','r',' ','low','e','s','t']
    Most frequent pair: ('low','e') → merge into 'lowe'
    Step 3: ['lowe',' ','lower',' ','lowest']
    ...and so on!
"""


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer built from scratch.
    This is a simplified version of what GPT uses!
    """

    def __init__(self):
        self.merges = {}        # (pair) → new_token
        self.vocab = {}         # token_id → token_string
        self.inverse_vocab = {} # token_string → token_id
        self.num_merges = 0

    def _get_pairs(self, tokens):
        """Count frequency of adjacent token pairs."""
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def _merge_pair(self, tokens, pair, new_token):
        """Replace all occurrences of pair with new_token."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(new_token)
                i += 2  # Skip both tokens in the pair
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, text, num_merges=50):
        """
        Train the BPE tokenizer on text.

        Args:
            text: Training text
            num_merges: How many merge operations to learn
                       (more merges = bigger vocabulary = longer tokens)
        """
        self.num_merges = num_merges

        # Start with individual characters as tokens
        # Each unique character gets an ID
        chars = sorted(set(text))
        self.vocab = {i: ch for i, ch in enumerate(chars)}
        self.inverse_vocab = {ch: i for i, ch in enumerate(chars)}
        next_id = len(chars)

        # Convert text to initial token IDs (character-level)
        tokens = [self.inverse_vocab[ch] for ch in text]

        print(f"  Starting vocabulary size: {len(self.vocab)}")
        print(f"  Starting token count: {len(tokens)}")
        print(f"  Learning {num_merges} merges...\n")

        # BPE: repeatedly merge the most frequent pair
        for step in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break

            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            best_count = pairs[best_pair]

            if best_count < 2:
                print(f"  Stopping early at step {step}: no pair appears 2+ times")
                break

            # Create new merged token
            new_token_str = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[next_id] = new_token_str
            self.inverse_vocab[new_token_str] = next_id
            self.merges[best_pair] = next_id

            # Apply the merge
            old_len = len(tokens)
            tokens = self._merge_pair(tokens, best_pair, next_id)

            if step < 10 or step % 10 == 0:
                print(f"  Merge {step:3d}: '{self.vocab[best_pair[0]]}' + "
                      f"'{self.vocab[best_pair[1]]}' → '{new_token_str}' "
                      f"(count={best_count}, tokens: {old_len}→{len(tokens)})")

            next_id += 1

        self.vocab_size = len(self.vocab)
        print(f"\n  Final vocabulary size: {self.vocab_size}")
        print(f"  Final token count: {len(tokens)}")
        print(f"  Compression ratio: {len(text)/len(tokens):.2f}x")

    def encode(self, text):
        """Encode text using learned BPE merges."""
        # Start with character-level tokens
        tokens = [self.inverse_vocab[ch] for ch in text if ch in self.inverse_vocab]

        # Apply merges in the order they were learned
        for pair, new_id in self.merges.items():
            tokens = self._merge_pair(tokens, pair, new_id)

        return tokens

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        return "".join(self.vocab[t] for t in token_ids if t in self.vocab)

    def tokenize_visual(self, text):
        """Show the tokenization visually."""
        tokens = self.encode(text)
        parts = [self.vocab[t] for t in tokens]
        return tokens, parts


def demo_bpe():
    """Demonstrate BPE tokenization."""
    print("=" * 60)
    print("🔤 BPE Tokenizer — How GPT Tokenizes Text!")
    print("=" * 60)

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

    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=60)

    # Demo encoding
    print("\n" + "=" * 60)
    print("📝 Encoding Examples:")
    print("=" * 60)

    samples = ["the little cat", "in the garden", "butterflies"]
    for sample in samples:
        ids, parts = tokenizer.tokenize_visual(sample)
        print(f"\n  '{sample}'")
        print(f"  Tokens: {parts}")
        print(f"  IDs:    {ids}")
        print(f"  Count:  {len(ids)} tokens (was {len(sample)} characters)")

    return tokenizer


if __name__ == "__main__":
    demo_bpe()
