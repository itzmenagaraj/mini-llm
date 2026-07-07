class CharacterTokenizer:
    def __init__(self, text):
        # Get all unique characters from training text
        chars = sorted(list(set(text)))

        self.vocab_size = len(chars)

        # Character → integer ID
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # Integer ID → character
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        """
        Convert text into list of token IDs.
        Example: "Hi" -> [12, 5]
        """
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        """
        Convert list of token IDs back into text.
        Example: [12, 5] -> "Hi"
        """
        return "".join([self.itos[i] for i in ids])


if __name__ == "__main__":
    with open("data/train/training.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharacterTokenizer(text)

    print("Vocabulary size:", tokenizer.vocab_size)
    print("Characters:", tokenizer.stoi)

    sample = "Hello"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print("Original:", sample)
    print("Encoded:", encoded)
    print("Decoded:", decoded)