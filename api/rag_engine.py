import faiss
import json
import numpy as np
import openai
from embeddings.embedder import embed
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FAISS INDEX (agora absoluto)
INDEX_PATH = os.path.join(BASE_DIR, "..", "embeddings", "vector_store.faiss")
print(">>> Abrindo FAISS em:", os.path.abspath(INDEX_PATH))
INDEX = faiss.read_index(os.path.abspath(INDEX_PATH))

# METADATA (já corrigido)
META_PATH = os.path.join(BASE_DIR, "..", "embeddings", "metadata.json")
print(">>> Abrindo metadata em:", os.path.abspath(META_PATH))
META = json.load(open(META_PATH, "r", encoding="utf-8"))


def search(query, k=5):
    q = np.array([embed(query)]).astype("float32")
    D, I = INDEX.search(q, k)
    results = []
    for idx in I[0]:
        path = META[idx]["path"]
        txt = open(path, "r", encoding="utf-8").read()
        results.append(txt)
    return results


def answer(query):
    context_chunks = search(query)
    context = "\n\n".join(context_chunks)

    prompt = f"""
Você é uma IA especializada em atas de RPPS.
Responda APENAS com base nos documentos abaixo.

[DOCUMENTOS]
{context}

[PERGUNTA]
{query}
"""

    # ---------------------------
    # TRATAMENTO DE ERRO OPENAI
    # ---------------------------
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você responde somente com base nos documentos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600
        )

        return resp.choices[0].message.content.strip()

    except Exception as e:
        # Isso impede o 500 e retorna msg amigável pro usuário final
        return f"⚠ Erro ao consultar o modelo da OpenAI: {str(e)}"
