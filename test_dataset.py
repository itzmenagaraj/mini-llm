from tokenizer import CharacterTokenizer
from dataset import TextDataset

with open("data/training.txt", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharacterTokenizer(text)

tokens = tokenizer.encode(text)

dataset = TextDataset(tokens, block_size=8)

x, y = dataset[0]

print("Input IDs :", x.tolist())
print("Target IDs:", y.tolist())

print()
print("Input Text :", tokenizer.decode(x.tolist()))
print("Target Text:", tokenizer.decode(y.tolist()))