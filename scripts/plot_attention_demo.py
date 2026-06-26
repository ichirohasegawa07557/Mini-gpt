from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import torch

from src.train import load_checkpoint
from src.visualize import plot_attention_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/mini_gpt_checkpoint.pt")
    parser.add_argument("--prompt", default="The robot learned")
    parser.add_argument("--out", default="results/attention_map.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_checkpoint(args.checkpoint, device=device)
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        model(ids)
    attention = model.blocks[0].attn.last_attention
    if attention is None:
        raise RuntimeError("No attention map captured.")
    plot_attention_map(attention, args.out)
    print(f"Saved attention map to {args.out}")


if __name__ == "__main__":
    main()
