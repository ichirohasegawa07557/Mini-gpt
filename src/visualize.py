from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch


def plot_training_history(history_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(history_csv)
    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["train_loss"], marker="o", label="train")
    plt.plot(df["step"], df["val_loss"], marker="o", label="val")
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Mini GPT training curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_attention_map(attention: torch.Tensor, output_path: str | Path, head: int = 0) -> None:
    if attention.ndim != 4:
        raise ValueError("Expected attention shape [batch, heads, T, T].")
    att = attention[0, head].detach().cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(att, aspect="auto")
    plt.colorbar(label="attention weight")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.title(f"Causal attention map, head={head}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
