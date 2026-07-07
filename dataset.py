import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, token_ids, block_size):

        self.data = token_ids
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):

        x = self.data[idx:idx + self.block_size]

        y = self.data[idx + 1:idx + self.block_size + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )