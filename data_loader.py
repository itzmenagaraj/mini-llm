from pathlib import Path


def load_training_text(data_dir="data/train"):
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    text_files = sorted(data_path.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    all_text = []

    for file_path in text_files:
        print(f"Loading: {file_path}")
        content = file_path.read_text(encoding="utf-8")
        all_text.append(content)

    return "\n\n".join(all_text)