import os
import json
import faiss
import numpy as np
from pathlib import Path
from embedder import embed

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

TXT_ROOT = "data/processed_txt/investimentos"
META_OUT = "embeddings/metadata.json"
INDEX_OUT = "embeddings/vector_store.faiss"

os.makedirs("embeddings", exist_ok=True)

# --------------------------------------------------
# UTIL
# --------------------------------------------------

def list_txts():
    for root, _, files in os.walk(TXT_ROOT):
        for f in files:
            if f.endswith(".txt"):
                yield os.path.join(root, f)

# --------------------------------------------------
# BUILD
# --------------------------------------------------

def build():
    metadata = []
    vectors = []

    txt_files = list(list_txts())
    print(f"📄 TXT de investimentos encontrados: {len(txt_files)}")

    for i, path in enumerate(txt_files, start=1):
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")

            if not text or len(text) < 50:
                continue

            # corta textos gigantes
            text = text[:6000]

            vec = embed(text)

            if not isinstance(vec, (list, np.ndarray)) or len(vec) != 768:
                continue

            vectors.append(vec)

            metadata.append({
                "path": path,
                "text": text[:2000]  # trecho para análise
            })

            if i % 200 == 0:
                print(f"🔄 Processados {i}/{len(txt_files)}")

        except Exception as e:
            print(f"[ERRO] {path}: {e}")

    if not vectors:
        raise RuntimeError("❌ Nenhum embedding válido foi gerado.")

    arr = np.array(vectors).astype("float32")

    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)

    faiss.write_index(index, INDEX_OUT)

    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("🎉 Index de investimentos criado com sucesso!")
    print(f"📦 FAISS: {INDEX_OUT}")
    print(f"📝 Metadata: {META_OUT}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    build()
