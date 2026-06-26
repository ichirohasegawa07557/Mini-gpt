from __future__ import annotations

from pathlib import Path
import streamlit as st
import torch

from src.train import load_checkpoint

st.set_page_config(page_title="Mini GPT From Scratch", layout="wide")
st.title("Mini GPT From Scratch")
st.write("Character-level GPT-style Transformer trained on a tiny local corpus.")

checkpoint = Path("results/mini_gpt_checkpoint.pt")
if not checkpoint.exists():
    st.warning("No checkpoint found. Run `python scripts/run_all.py` first.")
else:
    model, tokenizer = load_checkpoint(checkpoint, device="cpu")
    st.metric("Parameters", f"{model.parameter_count():,}")
    prompt = st.text_input("Prompt", "The ")
    max_new_tokens = st.slider("New tokens", 20, 300, 120)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.9)
    if st.button("Generate"):
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=20)[0]
        st.text_area("Generated text", tokenizer.decode(out), height=240)

    for image in ["results/training_curve.png", "results/attention_map.png"]:
        if Path(image).exists():
            st.image(image, caption=image)
