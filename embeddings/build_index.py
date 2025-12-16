import os
import json
import faiss
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from embedder import embed

TXT_ROOT = "data/processed_txt"
META_OUT = "embeddings/metadata.json"
INDEX_OUT = "embeddings/vector_store.faiss"

os.makedirs("embeddings", exist_ok=True)

# ---------- UTILIDADES ----------

def list_txts():
    for root, _, files in os.walk(TXT_ROOT):
        for f in files:
            if f.endswith(".txt"):
                yield os.path.join(root, f)

def extract_date_from_path(path: str):
    m = re.search(r"(20\d{2})[^\d]?(\d{2})?", path)
    if not m:
        return None
    year = m.group(1)
    month = m.group(2) if m.group(2) else "01"
    return f"{year}-{month}"

def extract_rpps_from_path(path: str):
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 4:
        return parts[3]
    return None

def extract_gestor_from_text(text: str):
    patterns = [
        r"gestor[a]?:\s*(.+)",
        r"gest[aã]o\s+(?:realizada|exercida)\s+por\s+(.+)",
        r"administrad[oa]\s+por\s+(.+)"
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()[:120]
    return None

# ---------- BUILD ----------

def build():
    metadata = []
    vectors = []

    txt_files = list(list_txts())
    print(f"\n📄 TXT encontrados: {len(txt_files)}\n")

    for path in tqdm(txt_files, desc="🔎 Indexando atas"):
        try:
            content = Path(path).read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()

            if len(content) < 50:
                continue

            date = extract_date_from_path(path)
            rpps = extract_rpps_from_path(path)
            gestor = extract_gestor_from_text(content)

            # ⚡ OPÇÃO 1: embedding único por documento
            text_for_embedding = content[:4000]

            vec = embed(text_for_embedding)
            if not isinstance(vec, (list, np.ndarray)) or len(vec) != 768:
                continue

            vectors.append(vec)
            metadata.append({
                "path": path.replace("\\", "/"),
                "date": date,
                "rpps": rpps,
                "gestor": gestor,
                "text": content[:8000]  # contexto rico para resumo
            })

        except Exception as e:
            print(f"[ERRO] {path}: {e}")

    if not vectors:
        print("\n❌ Nenhum embedding gerado.")
        return

    print(f"\n🧠 Total de documentos indexados: {len(vectors)}")

    arr = np.array(vectors).astype("float32")
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)

    faiss.write_index(index, INDEX_OUT)
    Path(META_OUT).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n🎉 Index criado com sucesso!")
    print(f"📦 FAISS: {INDEX_OUT}")
    print(f"📝 Metadata: {META_OUT}")
    print("✅ Build finalizado.\n")

if __name__ == "__main__":
    build()
