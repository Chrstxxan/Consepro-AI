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

# ==================================================
# 🔍 HEURÍSTICAS DE EXTRAÇÃO (SEM METADATA)
# ==================================================

def infer_rpps_from_text(text: str):
    patterns = [
        r"\bIPRE[A-Z]+\b",
        r"\bIPREM[A-Z]+\b",
        r"\bInstituto de Previdência(?: do| da)? ([A-ZÁ-Ú][A-Za-zÁ-Úãõç\s]+)",
        r"\bRPPS(?: do| da)? ([A-ZÁ-Ú][A-Za-zÁ-Úãõç\s]+)"
    ]

    found = set()
    for p in patterns:
        for m in re.findall(p, text, flags=re.IGNORECASE):
            if isinstance(m, tuple):
                found.add(m[0].strip())
            else:
                found.add(m.strip())

    return sorted(found)

def infer_date_from_text(text: str):
    m = re.search(
        r"(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+(20\d{2})",
        text.lower()
    )
    if m:
        return f"{m.group(1)} de {m.group(2)}"

    m = re.search(r"(20\d{2})", text)
    return m.group(1) if m else "data não identificada"

# ==================================================
# 🔎 DETECTORES DE INTENÇÃO
# ==================================================

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
        "existe processo",
        "há processo",
        "seleção de gestores",
        "renda fixa",
        "macro alocação"
    ])

# ==================================================
# 🔎 BUSCA SEMÂNTICA
# ==================================================

def semantic_search(query, k=40):
    q = np.array([embed(query)]).astype("float32")
    _, I = INDEX.search(q, k)
    return [META[i] for i in I[0]]

# ==================================================
# 🧠 ANSWER
# ==================================================

def answer(query: str) -> str:
    ql = query.lower()

    # --------------------------------------------------
    # 🔹 MODO ANALÍTICO (RENDA FIXA, PROCESSOS, ETC)
    # --------------------------------------------------
    if is_analytical_query(ql):

        expansion = """
        renda fixa títulos públicos tesouro nacional pré-fixado pós-fixado
        NTN LTN LFT CDI fundos conservadores
        análise estudo proposta deliberação aprovação apresentação
        gestores credenciamento performance
        """

        faiss_docs = semantic_search(expansion + " " + query)

        keywords = [
            "renda fixa", "títulos", "tesouro", "ntn", "ltn", "lft",
            "alocação", "deliberação", "aprovação", "estudo",
            "gestor", "gestores", "credenciamento", "performance"
        ]

        selected = []
        for d in faiss_docs:
            text = d.get("text", "").lower()
            if any(k in text for k in keywords):
                selected.append(d)

        if not selected:
            # 🔥 fallback semântico (NUNCA responder vazio)
            selected = faiss_docs[:8]

        context_blocks = []
        for d in selected:
            text = d.get("text", "")
            rpps = infer_rpps_from_text(text)
            date = infer_date_from_text(text)

            header = f"[Entidade(s): {', '.join(rpps) if rpps else 'não identificada'} | Data: {date}]"
            context_blocks.append(header + "\n" + text[:2500])

        context = "\n\n".join(context_blocks)

        prompt = f"""
Você é um analista sênior especializado em RPPS.

Objetivo:
- Identificar entidades que demonstrem interesse, análise, estudo,
  discussão ou deliberação relacionada ao tema da pergunta.
- NÃO exija decisão formal explícita.
- Use indícios, análises e menções contextuais.
- Se não houver decisão, explique o estágio (ex: estudo, análise, debate).

DOCUMENTOS:
{context}

PERGUNTA:
{query}

Responda listando as entidades em tópicos e explicando o contexto.
Não invente informações.
"""

        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Analise exclusivamente os documentos fornecidos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=900
        )

        return resp.choices[0].message.content.strip()

    # --------------------------------------------------
    # 🔹 RESUMOS
    # --------------------------------------------------
    if is_summary_query(ql):
        docs = semantic_search(query, k=12)

        context = "\n\n".join(
            d.get("text", "")[:3000] for d in docs
        )

        prompt = f"""
Você é um analista especializado em RPPS.

Tarefa:
- Elaborar um resumo executivo claro e objetivo
- Destacar decisões, análises, estudos e encaminhamentos
- Agrupar informações semelhantes

DOCUMENTOS:
{context}

PERGUNTA:
{query}
"""

        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Resuma apenas com base nos documentos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )

        return resp.choices[0].message.content.strip()

    # --------------------------------------------------
    # 🔹 BUSCA PADRÃO
    # --------------------------------------------------
    docs = semantic_search(query, k=8)

    context = "\n\n".join(
        d.get("text", "")[:2500] for d in docs
    )

    prompt = f"""
Você é um analista especializado em atas de RPPS.

Regras:
- Use apenas as informações fornecidas
- Seja objetivo
- Não invente dados

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
