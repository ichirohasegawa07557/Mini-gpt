from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.train import train_char_model, load_checkpoint
from src.visualize import plot_training_history, plot_attention_map


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("Training a tiny MiniGPT demo...", flush=True)
    train_char_model(
        text_path="data/tiny_corpus.txt",
        out_dir=out_dir,
        max_iters=1,
        batch_size=4,
        block_size=32,
        n_layer=1,
        n_head=1,
        n_embd=16,
        eval_iters=1,
    )
    plot_training_history(out_dir / "training_history.csv", out_dir / "training_curve.png")

    print("Generating text...", flush=True)
    model, tokenizer = load_checkpoint(out_dir / "mini_gpt_checkpoint.pt", device="cpu")
    prompt = "The "
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=5, temperature=0.0, top_k=None)[0]
    generated = tokenizer.decode(out)
    (out_dir / "generated_text.txt").write_text(generated, encoding="utf-8")

    print("Plotting attention map...", flush=True)
    prompt_ids = torch.tensor([tokenizer.encode("The robot learned")], dtype=torch.long)
    with torch.no_grad():
        model(prompt_ids)
    attention = model.blocks[0].attn.last_attention
    if attention is not None:
        plot_attention_map(attention, out_dir / "attention_map.png")

    print("Done. Results saved in results/.", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
