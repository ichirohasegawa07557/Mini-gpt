from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return "".join(self.itos[int(i)] for i in ids)


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_dataset(text: str) -> tuple[CharTokenizer, torch.Tensor]:
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return tokenizer, data


def train_val_split(data: torch.Tensor, split: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(len(data) * split)
    return data[:n], data[n:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError("Dataset is too short for the requested block_size.")
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y
