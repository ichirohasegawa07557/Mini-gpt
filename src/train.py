from __future__ import annotations

from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from src.data import build_dataset, get_batch, load_text, train_val_split
from src.model import GPTConfig, MiniGPT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def estimate_loss(model: MiniGPT, data: torch.Tensor, batch_size: int, block_size: int, eval_iters: int, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def train_char_model(
    text_path: str | Path,
    out_dir: str | Path = "results",
    max_iters: int = 200,
    batch_size: int = 16,
    block_size: int = 64,
    n_layer: int = 2,
    n_head: int = 2,
    n_embd: int = 64,
    learning_rate: float = 3e-3,
    eval_interval: int = 25,
    eval_iters: int = 1,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, object]:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = load_text(text_path)
    tokenizer, data = build_dataset(text)
    train_data, val_data = train_val_split(data)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = MiniGPT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    history = []
    for step in range(max_iters + 1):
        if step % eval_interval == 0 or step == max_iters:
            train_loss = estimate_loss(model, train_data, batch_size, block_size, eval_iters, device)
            val_source = val_data if len(val_data) > block_size + 1 else train_data
            val_loss = estimate_loss(model, val_source, batch_size, block_size, eval_iters, device)
            history.append({"step": step, "train_loss": train_loss, "val_loss": val_loss})

        x, y = get_batch(train_data, batch_size, block_size, device)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    checkpoint = {
        "model_state": model.state_dict(),
        "config": config.__dict__,
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
    }
    torch.save(checkpoint, out_dir / "mini_gpt_checkpoint.pt")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    metadata = {
        "vocab_size": tokenizer.vocab_size,
        "parameter_count": model.parameter_count(),
        "max_iters": max_iters,
        "block_size": block_size,
        "device": device,
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"model": model, "tokenizer": tokenizer, "history": pd.DataFrame(history), "metadata": metadata}


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[MiniGPT, object]:
    from src.data import CharTokenizer

    ckpt = torch.load(path, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = MiniGPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    itos = {int(k): v for k, v in ckpt["itos"].items()} if isinstance(next(iter(ckpt["itos"].keys())), str) else ckpt["itos"]
    tokenizer = CharTokenizer(stoi=ckpt["stoi"], itos=itos)
    return model, tokenizer
