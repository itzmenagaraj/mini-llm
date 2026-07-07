# mini-llm

A lightweight implementation of a transformer-based language model with Byte-Pair Encoding (BPE) tokenization. This project demonstrates core concepts of modern LLMs including multi-head attention, feed-forward networks, and autoregressive text generation.

## Features

- **Transformer Architecture**: Multi-head self-attention mechanism with residual connections
- **BPE Tokenization**: Subword tokenization for efficient vocabulary management
- **Efficient Training**: GPU/CUDA support for faster training
- **Text Generation**: Autoregressive sampling with configurable temperature and context
- **Modular Design**: Clean separation of tokenization, model, and training logic

## Architecture

The model consists of:

- **Token & Position Embeddings**: Converts tokens and positions into dense vectors
- **Multi-Head Self-Attention**: 4 parallel attention heads for diverse feature representations
- **Feed-Forward Networks**: Position-wise fully connected layers with ReLU activation
- **Transformer Blocks**: 2 stacked blocks combining attention and feed-forward layers
- **Causal Masking**: Ensures the model only attends to previous tokens

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BLOCK_SIZE` | 64 | Maximum sequence length |
| `N_EMBED` | 128 | Embedding dimension |
| `N_HEAD` | 4 | Number of attention heads |
| `N_LAYER` | 2 | Number of transformer blocks |
| `VOCAB_SIZE` | 300 | BPE vocabulary size |
| `BATCH_SIZE` | 32 | Training batch size |
| `MAX_ITERS` | 3000 | Maximum training iterations |

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/itzmenagaraj/mini-llm.git
cd mini-llm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train the model on your text data:

```bash
python train.py
```

The training script will:
1. Load text from `data/train/` directory
2. Train a BPE tokenizer (vocab_size=300)
3. Create a TextDataset with block_size=64
4. Train the MiniGPT model for 3000 iterations
5. Save the model to `models/mini_gpt.pth`
6. Save the tokenizer to `models/bpe_tokenizer.json`

### Text Generation

Generate text using the trained model:

```bash
python generate.py --prompt "Your starting text here" --length 50
```

Example:
```bash
python generate.py --prompt "The future of AI" --length 100
```

The generation script will:
- Load the trained model and tokenizer
- Start with your prompt
- Autoregressively generate tokens
- Display top-5 token predictions at each step
- Output the complete generated text

## Project Structure

```
mini-llm/
├── model.py              # Core transformer model architecture
├── train.py              # Training loop
├── generate.py           # Text generation script
├── config.py             # Hyperparameter configuration
├── bpe_tokenizer.py      # BPE tokenizer implementation
├── tokenizer.py          # Character-level tokenizer
├── dataset.py            # PyTorch Dataset class
├── data_loader.py        # Data loading utilities
├── requirements.txt      # Project dependencies
├── data/                 # Training data
│   ├── books/
│   ├── manuals/
│   ├── procedures/
│   ├── root_causes/
│   ├── tickets/
│   └── train/
│       └── training.txt
└── models/               # Model checkpoints
    ├── mini_gpt.pth      # Trained model weights
    └── bpe_tokenizer.json # BPE tokenizer vocabulary
```

## Configuration

Edit `config.py` to adjust hyperparameters:

```python
BLOCK_SIZE = 64          # Sequence length (increase for longer context)
BATCH_SIZE = 32          # Batch size (increase for faster training if GPU memory allows)
N_EMBED = 128            # Embedding dimension (larger = more capacity)
N_HEAD = 4               # Attention heads (must divide N_EMBED)
N_LAYER = 2              # Transformer layers (deeper = more capacity)
LEARNING_RATE = 1e-3     # Optimizer learning rate
MAX_ITERS = 3000         # Training iterations
VOCAB_SIZE = 300         # BPE vocabulary size
```

## Training Data

Place your training text files in the `data/train/` directory. The model will concatenate all text files in that directory for training. Supported formats:
- `.txt` files (UTF-8 encoded)

For best results:
- Use 50K+ tokens of training data
- Ensure consistent text quality
- Consider domain-specific text for specialized models

## Dependencies

- `torch`: Deep learning framework
- `numpy`: Numerical computing (implicit via torch)

## Model Performance

The model learns to:
- Capture character and subword patterns
- Generate contextually relevant completions
- Understand basic syntax and structure

Performance scales with:
- Training data size
- Model capacity (increase layers/embeddings)
- Training iterations

## Future Improvements

- [ ] Support for larger vocabularies and block sizes
- [ ] Temperature/top-k sampling for generation
- [ ] Validation set evaluation metrics
- [ ] Gradient checkpointing for memory efficiency
- [ ] Distributed training support
- [ ] Model quantization for inference

## Learning Resources

This implementation covers:
- Transformer architecture (Vaswani et al., 2017)
- Multi-head self-attention mechanisms
- Positional encoding
- Byte-Pair Encoding tokenization
- Autoregressive language modeling

## License

MIT License

## Author

Nagaraj Balan

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

---

## Getting Started Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (takes ~5-10 minutes on CPU, seconds on GPU)
python train.py

# 3. Generate text
python generate.py --prompt "Once upon a time" --length 100
```