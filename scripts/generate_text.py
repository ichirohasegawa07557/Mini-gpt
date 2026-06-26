from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import torch

from src.train import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/mini_gpt_checkpoint.pt")
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", default="results/generated_text.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_checkpoint(args.checkpoint, device=device)
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0]
    text = tokenizer.decode(out_ids)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved generated text to {args.out}")


if __name__ == "__main__":
    main()
