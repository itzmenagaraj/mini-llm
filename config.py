# Model configuration

BLOCK_SIZE = 64      # How many characters the model sees at once
BATCH_SIZE = 32      # How many examples per training batch

N_EMBED = 128        # Embedding vector size
N_HEAD = 4           # Number of attention heads
N_LAYER = 2          # Number of transformer blocks

DROPOUT = 0.2
LEARNING_RATE = 1e-3
MAX_ITERS = 3000
EVAL_INTERVAL = 300

MODEL_PATH = "models/mini_gpt.pth"

TOKENIZER_PATH = "models/bpe_tokenizer.json"
VOCAB_SIZE = 300