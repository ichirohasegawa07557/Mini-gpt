from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.train import train_char_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="data/tiny_corpus.txt")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--eval-iters", type=int, default=1)
    args = parser.parse_args()
    result = train_char_model(
        text_path=args.text,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        eval_iters=args.eval_iters,
    )
    out_dir = Path(args.out_dir)
    from src.visualize import plot_training_history
    plot_training_history(out_dir / "training_history.csv", out_dir / "training_curve.png")
    print("Training complete.")
    print(result["metadata"])
    print(f"Saved checkpoint to {out_dir / 'mini_gpt_checkpoint.pt'}")


if __name__ == "__main__":
    main()
