import os
import json
import faiss
import numpy as np
import re
import openai
from embeddings.embedder import embed

openai.api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "..", "embeddings", "vector_store.faiss")
META_PATH = os.path.join(BASE_DIR, "..", "embeddings", "metadata.json")

INDEX = faiss.read_index(INDEX_PATH)
META = json.load(open(META_PATH, encoding="utf-8"))

# --------------------------------------------------
# DETECTORES
# --------------------------------------------------

def is_summary_query(q: str) -> bool:
    q = q.lower()
    return any(x in q for x in ["resuma", "resumo", "síntese", "principais pontos", "panorama"])

def is_analytical_query(q: str) -> bool:
    q = q.lower()
    return any(x in q for x in [
        "quais entidades",
        "quem está",
        "quem pretende",
        "quem quer",
        "tendência",
        "panorama"
    ])

def extract_date_from_query(q: str):
    m = re.search(r"(20\d{2})[^\d]?(\d{2})?", q)
    if not m:
        return None
    year = m.group(1)
    month = m.group(2)
    return f"{year}-{month}" if month else year

def extract_gestor_from_query(q: str):
    m = re.search(r"gestor\s+(.+)", q.lower())
    if m:
        return m.group(1).strip()
    return None

# --------------------------------------------------
# CONSULTAS ESTRUTURAIS
# --------------------------------------------------

def rpps_by_gestor(nome):
    return sorted({
        m["rpps"]
        for m in META
        if m.get("gestor") and nome.lower() in m["gestor"].lower()
    })

def docs_by_date(target):
    return [
        m for m in META
        if m.get("date") and m["date"].startswith(target)
    ]

# --------------------------------------------------
# BUSCA SEMÂNTICA
# --------------------------------------------------

def semantic_search(query, k=8):
    q = np.array([embed(query)]).astype("float32")
    _, I = INDEX.search(q, k)
    return [META[i] for i in I[0]]

# --------------------------------------------------
# SELEÇÃO DIVERSIFICADA POR RPPS (🔥 CORREÇÃO DO VIÉS)
# --------------------------------------------------

def select_diverse_docs(docs, max_per_rpps=3, max_total=40):
    by_rpps = {}

    for d in docs:
        rpps = d.get("rpps")
        if not rpps:
            continue
        by_rpps.setdefault(rpps, []).append(d)

    # prioriza documentos mais recentes por RPPS
    for rpps in by_rpps:
        by_rpps[rpps].sort(
            key=lambda x: x.get("date") or "",
            reverse=True
        )

    selected = []
    for rpps_docs in by_rpps.values():
        selected.extend(rpps_docs[:max_per_rpps])

    return selected[:max_total]

# --------------------------------------------------
# ANSWER
# --------------------------------------------------

def answer(query: str) -> str:
    ql = query.lower()

    # ------------------------------
    # 1️⃣ RPPS por gestor
    # ------------------------------
    gestor = extract_gestor_from_query(ql)
    if gestor:
        rpps = rpps_by_gestor(gestor)
        if rpps:
            return (
                f"Os RPPS que utilizam o gestor {gestor} são:\n- "
                + "\n- ".join(rpps)
            )
        return f"Não foi encontrado RPPS associado ao gestor {gestor}."

    # ------------------------------
    # 2️⃣ MODO ANALÍTICO (ex: renda fixa)
    # ------------------------------
    if is_analytical_query(ql):
        keywords = []

        if "renda fixa" in ql:
            keywords.append("renda fixa")

        if keywords:
            raw_docs = [
                m for m in META
                if any(k in m.get("text", "").lower() for k in keywords)
            ]

            docs = select_diverse_docs(raw_docs)

            if not docs:
                return "Não foram encontradas atas relevantes para essa análise."

            context = "\n\n".join(
                f"[RPPS: {d.get('rpps')} | Data: {d.get('date')}]\n{d.get('text')}"
                for d in docs
            )

            prompt = f"""
Você é um analista financeiro especializado em RPPS.

Com base nas atas abaixo:
- identifique entidades que demonstrem interesse, intenção ou deliberação
  relacionada ao tema consultado.
- considere estudos, propostas, aprovações ou discussões.
- se não houver evidência clara, deixe isso explícito.

DOCUMENTOS:
{context}

PERGUNTA:
{query}

Responda listando as entidades e explicando brevemente o contexto.
Não invente informações.
"""

            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Analise apenas com base nos documentos fornecidos."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=700
            )

            return resp.choices[0].message.content.strip()

    # ------------------------------
    # 3️⃣ RESUMO POR DATA
    # ------------------------------
    date = extract_date_from_query(ql)
    if date and is_summary_query(ql):
        docs = docs_by_date(date)
        if not docs:
            return f"Não há atas disponíveis para o período {date}."

        docs = select_diverse_docs(docs, max_per_rpps=2, max_total=30)

        context = "\n\n".join(
            f"[RPPS: {d.get('rpps')} | Data: {d.get('date')}]\n{d.get('text')}"
            for d in docs
        )

    # ------------------------------
    # 4️⃣ BUSCA SEMÂNTICA NORMAL
    # ------------------------------
    else:
        docs = semantic_search(query)
        docs = select_diverse_docs(docs, max_per_rpps=1, max_total=8)

        context = "\n\n".join(
            f"[RPPS: {d.get('rpps')} | Data: {d.get('date')}]\n{d.get('text')}"
            for d in docs
        )

    prompt = f"""
Você é um analista especializado em atas de RPPS.

Regras:
- Use somente as informações fornecidas
- Inferir intenções apenas quando houver indícios claros
- Não inventar dados
- Responder de forma objetiva e profissional

DOCUMENTOS:
{context}

PERGUNTA:
{query}
"""

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Responda com base nos documentos."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700
    )

    return resp.choices[0].message.content.strip()
