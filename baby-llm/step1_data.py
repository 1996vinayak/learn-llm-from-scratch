"""
╔══════════════════════════════════════════════════════════════╗
║           STEP 1: TRAINING DATA  📖                         ║
║           "The Recipe Book for Our Baby LLM"                 ║
╚══════════════════════════════════════════════════════════════╝

🎯 WHAT IS TRAINING DATA?
========================
Training data is the TEXT that our model will learn from.
Think of it like a recipe book — the model reads it over and over
until it learns the patterns.

Real LLMs like ChatGPT are trained on:
  - Books, websites, Wikipedia, code, etc.
  - Hundreds of BILLIONS of words!

Our Baby LLM will learn from a small story.
It's tiny, but the CONCEPT is identical!

🤔 WHY DOES THIS MATTER?
========================
"Garbage in, garbage out" — if you train on bad data, you get bad results.
If you train on Shakespeare, the model writes like Shakespeare.
If you train on code, the model writes code.
We'll use a simple repetitive story so our tiny model can learn patterns.
"""


def get_training_data():
    """
    Returns the text our model will learn from.

    We use a simple, repetitive story because:
    1. Our model is TINY — it needs simple patterns
    2. Repetition helps learning (just like kids learn by repetition!)
    3. We can easily see if the model learned something

    In real LLMs, this would be terabytes of text from the internet.
    """

    text = """
    Once upon a time there was a little cat.
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
    The happy little cat lived in a beautiful garden.
    """

    return text


def explore_data(text):
    """
    Let's look at our data and understand it!

    This is called "Exploratory Data Analysis" (EDA).
    Real ML engineers ALWAYS look at their data first.
    """

    print("=" * 60)
    print("📖 STEP 1: Exploring Our Training Data")
    print("=" * 60)

    # How much text do we have?
    print(f"\n📏 Total characters in our text: {len(text)}")
    print(f"📝 Total words: {len(text.split())}")
    print(f"📄 Total lines: {len(text.strip().split(chr(10)))}")

    # What unique characters does our text use?
    unique_chars = sorted(set(text))
    print(f"\n🔤 Unique characters: {len(unique_chars)}")
    print(f"   They are: {unique_chars}")

    # Show a preview
    print(f"\n👀 First 200 characters of our text:")
    print(f"   '{text[:200]}...'")

    print("\n" + "=" * 60)
    print("✅ Data is ready! Our model will learn from this story.")
    print("=" * 60)

    return unique_chars


# ═══════════════════════════════════════════════════════════
# 🏃 RUN THIS FILE to see the data exploration
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    text = get_training_data()
    explore_data(text)
