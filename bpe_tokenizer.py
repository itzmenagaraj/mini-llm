import json
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, vocab_size=300):
        tokens = list(text)

        self.vocab = {ch: i for i, ch in enumerate(sorted(set(tokens)))}

        while len(self.vocab) < vocab_size:
            pairs = Counter(zip(tokens, tokens[1:]))

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_token = "".join(best_pair)

            if new_token in self.vocab:
                break

            self.merges[best_pair] = new_token
            self.vocab[new_token] = len(self.vocab)

            tokens = self._merge_tokens(tokens, best_pair, new_token)

        self.id_to_token = {i: tok for tok, i in self.vocab.items()}

    def _merge_tokens(self, tokens, pair, new_token):
        result = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                result.append(new_token)
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        return result

    def encode(self, text):
        tokens = list(text)

        for pair, new_token in self.merges.items():
            tokens = self._merge_tokens(tokens, pair, new_token)

        return [self.vocab[tok] for tok in tokens]

    def decode(self, ids):
        return "".join([self.id_to_token[i] for i in ids])

    def save(self, path):
        data = {
            "vocab": self.vocab,
            "merges": [[a, b, c] for (a, b), c in self.merges.items()]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        self.merges = {(a, b): c for a, b, c in data["merges"]}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}