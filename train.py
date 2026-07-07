import torch
from torch.utils.data import DataLoader

from bpe_tokenizer import BPETokenizer

from dataset import TextDataset
from model import MiniGPT
from config import BLOCK_SIZE, BATCH_SIZE, LEARNING_RATE, MAX_ITERS, MODEL_PATH
from data_loader import load_training_text


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

text = load_training_text("data/train")

tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=300)
tokenizer.save("models/bpe_tokenizer.json")

tokens = tokenizer.encode(text)

print("Total tokens:", len(tokens))
print("Block size:", BLOCK_SIZE)

if len(tokens) <= BLOCK_SIZE:
    raise ValueError(
        f"Training data too small. Total tokens={len(tokens)}, BLOCK_SIZE={BLOCK_SIZE}. "
        "Add more text or reduce BLOCK_SIZE."
    )

dataset = TextDataset(tokens, BLOCK_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MiniGPT(tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(MAX_ITERS):
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": len(tokenizer.vocab),
    "block_size": BLOCK_SIZE,
}, MODEL_PATH)

print("Model saved:", MODEL_PATH)