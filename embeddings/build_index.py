import os
import json
import faiss
import numpy as np
import re
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
# EXTRAÇÃO E HEURÍSTICAS
# --------------------------------------------------

def extract_rpps(text: str, path: str):
    patterns = [
        r"\bIPRE[A-Z]+\b",
        r"\bIPREM[A-Z]+\b",
        r"\bIPREV[A-Z]+\b",
        r"\bINPRE[A-Z]+\b"
    ]

    found = set()
    for p in patterns:
        found.update(re.findall(p, text.upper()))

    if found:
        return sorted(found)

    # fallback pelo path
    for part in path.split(os.sep):
        if part.upper().startswith(("IPRE", "IPREM", "IPREV", "INPRE")):
            return [part.upper()]

    return []

def extract_date(text: str):
    month_map = {
        "janeiro": 1, "fevereiro": 2, "março": 3,
        "abril": 4, "maio": 5, "junho": 6,
        "julho": 7, "agosto": 8, "setembro": 9,
        "outubro": 10, "novembro": 11, "dezembro": 12
    }

    m = re.search(
        r"(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+(20\d{2})",
        text.lower()
    )

    if m:
        return {
            "ano": int(m.group(2)),
            "mes": month_map[m.group(1)]
        }

    m = re.search(r"(20\d{2})", text)
    return {"ano": int(m.group(1)), "mes": None} if m else {"ano": None, "mes": None}

def classify_document(text: str):
    t = text.lower()

    if "comitê de investimentos" in t or "comite de investimentos" in t:
        return "comite_investimentos"

    if "política de investimentos" in t:
        return "politica_investimentos"

    if "conselho fiscal" in t:
        return "conselho_fiscal"

    if "conselho deliberativo" in t:
        return "conselho_deliberativo"

    return "outros"

def semantic_flags(text: str):
    t = text.lower()
    return {
        "tem_investimentos": any(k in t for k in ["investimento", "aplicação", "alocação"]),
        "tem_renda_fixa": any(k in t for k in ["renda fixa", "tesouro", "ltn", "ntn", "lft"]),
        "tem_selecao_gestor": any(k in t for k in ["credenciamento", "seleção", "gestor"]),
        "tem_performance": any(k in t for k in ["rentabilidade", "performance", "resultado"]),
        "menciona_membros": any(k in t for k in ["presentes", "conselheiros", "membros"])
    }

# --------------------------------------------------
# LIST TXT
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

            if not text or len(text) < 100:
                continue

            text = text[:6000]

            vec = embed(text)
            if not isinstance(vec, (list, np.ndarray)) or len(vec) != 768:
                continue

            rpps = extract_rpps(text, path)
            date_info = extract_date(text)
            doc_type = classify_document(text)
            flags = semantic_flags(text)

            vectors.append(vec)

            metadata.append({
                "path": path,
                "text": text[:2500],
                "rpps": rpps,
                "orgao": doc_type,
                "ano": date_info["ano"],
                "mes": date_info["mes"],
                **flags
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

    print("🎉 Index reconstruído com metadata enriquecida!")
    print(f"📦 FAISS: {INDEX_OUT}")
    print(f"📝 Metadata: {META_OUT}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    build()
