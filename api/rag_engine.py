import faiss
import json
import numpy as np
import openai
from embeddings.embedder import embed
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

INDEX = faiss.read_index("embeddings/vector_store.faiss")
META = json.load(open("embeddings/metadata.json","r",encoding="utf-8"))

def search(query, k=5):
    q = np.array([embed(query)]).astype("float32")
    D, I = INDEX.search(q, k)
    results = []
    for idx in I[0]:
        path = META[idx]["path"]
        txt = open(path,"r",encoding="utf-8").read()
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

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você responde somente com base nos documentos."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600
    )

    return resp.choices[0].message.content.strip()
