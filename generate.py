import argparse
import torch

from model import MiniGPT
from bpe_tokenizer import BPETokenizer
from config import BLOCK_SIZE, MODEL_PATH, TOKENIZER_PATH


@torch.no_grad()
def generate(model, tokenizer, start_text, max_new_tokens=20):
    model.eval()

    print("\n========== GENERATION START ==========")
    print("Start text:", start_text)

    encoded = tokenizer.encode(start_text)
    print("Encoded IDs:", encoded)

    idx = torch.tensor([encoded], dtype=torch.long)
    print("Initial tensor shape:", idx.shape)

    for step in range(max_new_tokens):

        print("\n--------------------------------------")
        print(f"STEP {step + 1}")

        idx_cond = idx[:, -BLOCK_SIZE:]

        print("Current text:", tokenizer.decode(idx[0].tolist()))
        print("Input IDs to model:", idx_cond[0].tolist())
        print("Input tensor shape:", idx_cond.shape)

        logits, _ = model(idx_cond)

        print("Logits shape:", logits.shape)

        last_logits = logits[:, -1, :]
        print("Last token logits shape:", last_logits.shape)

        probs = torch.softmax(last_logits, dim=-1)
        print("Probabilities shape:", probs.shape)

        top_probs, top_ids = torch.topk(probs, k=5)

        print("\nTop 5 predictions")

        for i in range(5):
            token_id = top_ids[0][i].item()
            token_prob = top_probs[0][i].item()
            token_text = tokenizer.decode([token_id])

            print(
                f"{i+1}. Token={repr(token_text)} "
                f"ID={token_id} "
                f"Probability={token_prob:.6f}"
            )

        next_id = torch.multinomial(probs, num_samples=1)

        selected_id = next_id.item()
        selected_token = tokenizer.decode([selected_id])

        print(f"\nSelected Token : {repr(selected_token)}")
        print(f"Selected ID    : {selected_id}")

        idx = torch.cat((idx, next_id), dim=1)

    print("\n========== GENERATION END ==========")

    final_text = tokenizer.decode(idx[0].tolist())

    print("\nGenerated Text\n")
    print(final_text)

    return final_text


def main():

    parser = argparse.ArgumentParser(description="MiniGPT Generator")

    parser.add_argument(
        "--prompt",
        required=True,
        type=str,
        help="Prompt to start generation"
    )

    parser.add_argument(
        "--max_tokens",
        default=20,
        type=int,
        help="Maximum new tokens"
    )

    args = parser.parse_args()

    print("Loading checkpoint...")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    print("Loading tokenizer...")

    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_PATH)

    print("Loading model...")

    model = MiniGPT(checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Model Loaded Successfully.\n")

    generate(
        model=model,
        tokenizer=tokenizer,
        start_text=args.prompt,
        max_new_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()