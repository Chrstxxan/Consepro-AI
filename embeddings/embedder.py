# embedder.py — versão OFFLINE/PT com E5-large
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-base"

print(f"[EMBEDDER] Carregando modelo {MODEL_NAME} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

@torch.no_grad()
def embed(text: str):
    encoded = tokenizer(
        text,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        model_output = model(**encoded)

    vec = model_output.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    # garante tamanho correto
    if vec.shape[0] != 768:
        raise ValueError(f"Embedding com shape inesperado: {vec.shape}")

    return vec.tolist()
