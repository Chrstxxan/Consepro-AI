import faiss
from pathlib import Path

p = Path("embeddings/vector_store.faiss")

print("Arquivo existe?", p.exists())

try:
    idx = faiss.read_index(str(p))
    print("FAISS carregado! Dimensão:", idx.d)
except Exception as e:
    print("Erro ao abrir o FAISS:", e)
