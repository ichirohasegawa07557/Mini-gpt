from __future__ import annotations

import torch

from src.data import build_dataset, get_batch
from src.model import GPTConfig, MiniGPT


def test_tokenizer_roundtrip() -> None:
    tok, data = build_dataset("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"
    assert len(data) == len("hello world")


def test_get_batch_shapes() -> None:
    _, data = build_dataset("abcdefghijklmnopqrstuvwxyz")
    x, y = get_batch(data, batch_size=4, block_size=8)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_model_forward_and_loss() -> None:
    config = GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = MiniGPT(config)
    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, 20)
    assert loss is not None
    assert torch.isfinite(loss)


def test_generate_extends_sequence() -> None:
    config = GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = MiniGPT(config)
    x = torch.randint(0, 20, (1, 4))
    out = model.generate(x, max_new_tokens=5)
    assert out.shape == (1, 9)
