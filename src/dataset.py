import torch
from torch.utils.data import Dataset, DataLoader


class KortexDataset(Dataset):
    def __init__(self, token_ids, seq_len=1024):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class StreamingKortexDataset:
    def __init__(self, tokenizer, seq_len=1024, max_length=4096):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_length = max_length
        self.buffer = []

    def _tokenize_text(self, text):
        ids = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
        return ids

    def add_text(self, text):
        ids = self._tokenize_text(text)
        self.buffer.extend(ids)

    def get_batch(self, batch_size):
        while len(self.buffer) < (self.seq_len + 1) * batch_size:
            return None

        xs, ys = [], []
        for _ in range(batch_size):
            start = torch.randint(0, max(1, len(self.buffer) - self.seq_len - 1), (1,)).item()
            chunk = self.buffer[start : start + self.seq_len + 1]
            if len(chunk) < self.seq_len + 1:
                continue
            xs.append(chunk[:-1])
            ys.append(chunk[1:])

        if len(xs) < batch_size:
            return None

        self.buffer = self.buffer[batch_size * self.seq_len // 2:]
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def get_data_loader(token_ids, batch_size=4, seq_len=1024, num_workers=0):
    dataset = KortexDataset(token_ids, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
