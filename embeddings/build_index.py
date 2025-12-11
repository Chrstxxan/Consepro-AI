import os
import json
import faiss
import numpy as np
from embedder import embed
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

TXT_ROOT = "processed_txt"
META_OUT = "embeddings/metadata.json"
INDEX_OUT = "embeddings/vector_store.faiss"

os.makedirs("embeddings", exist_ok=True)

indexed_state_file = "index_state.json"

def load_index_state():
    if Path(indexed_state_file).exists():
        return json.loads(Path(indexed_state_file).read_text())
    return {}

def save_index_state(state):
    Path(indexed_state_file).write_text(json.dumps(state, indent=2))

def list_txts():
    for root,_,files in os.walk(TXT_ROOT):
        for f in files:
            if f.endswith(".txt"):
                yield os.path.join(root,f)

def build():
    metadata = []
    vectors = []

    for path in list_txts():
        with open(path,"r",encoding="utf-8") as f:
            content = f.read()

        vec = embed(content)
        vectors.append(vec)
        metadata.append({"path": path})

    arr = np.array(vectors).astype("float32")

    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)

    faiss.write_index(index, INDEX_OUT)
    json.dump(metadata, open(META_OUT,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"🎉 Index construído com {len(metadata)} documentos!")

if __name__ == "__main__":
    build()
