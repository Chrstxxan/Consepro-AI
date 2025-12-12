import os
import json
import faiss
import numpy as np
from embedder import embed
from dotenv import load_dotenv
from pathlib import Path
import unicodedata

load_dotenv()

TXT_ROOT = "data/processed_txt"
META_OUT = "embeddings/metadata.json"
INDEX_OUT = "embeddings/vector_store.faiss"

os.makedirs("embeddings", exist_ok=True)

def list_txts():
    for root, _, files in os.walk(TXT_ROOT):
        for f in files:
            if f.endswith(".txt"):
                yield os.path.join(root, f)

def is_valid_vector(v):
    return (
        v is not None
        and isinstance(v, (list, np.ndarray))
        and len(v) > 0
    )

def fix_string(s):
    """Garante que strings terão apenas caracteres UTF-8 válidos."""
    return (
        unicodedata.normalize("NFC", s)
        .encode("utf-8", "replace")
        .decode("utf-8", "replace")
    )

def build():
    metadata = []
    vectors = []

    txt_files = list(list_txts())
    print(f" TXT encontrados: {len(txt_files)}")

    for path in txt_files:
        try:
            content = Path(path).read_text(encoding="utf-8")
            content = content[:5000]  # limita tamanho

            vec = embed(content)

            if not is_valid_vector(vec) or len(vec) != 768:
                print(f"[SKIP] Vetor inválido ou dimensão errada em {path}")
                continue

            vectors.append(vec)
            metadata.append({"path": fix_string(path)})

        except Exception as e:
            print(f"[ERRO] Falha ao processar {path}: {e}")
            continue

    if len(vectors) == 0:
        print(" ERRO: Nenhum embedding válido!")
        return

    arr = np.array(vectors).astype("float32")

    print(f" Dimensão dos vetores: {arr.shape}")

    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)

    faiss.write_index(index, INDEX_OUT)

    # salva metadata limpo
    clean_json = json.dumps(metadata, indent=2, ensure_ascii=False)
    Path(META_OUT).write_text(clean_json, encoding="utf-8")

    print(f" Index construído com {len(metadata)} documentos!")
    print(f" FAISS salvo em {INDEX_OUT}")
    print(f" Metadata salvo em {META_OUT}")

if __name__ == "__main__":
    build()
