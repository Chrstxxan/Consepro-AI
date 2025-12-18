import os
import json
import faiss
import numpy as np
import re
import random
import openai
from datetime import datetime
from embeddings.embedder import embed

# ==================================================
# 🔑 CONFIG
# ==================================================

openai.api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "embeddings", "vector_store.faiss")
META_PATH = os.path.join(BASE_DIR, "..", "embeddings", "metadata.json")

INDEX = faiss.read_index(INDEX_PATH)
META = json.load(open(META_PATH, encoding="utf-8"))

CURRENT_YEAR = datetime.now().year

# ==================================================
# 🔒 CONTROLES (ANTI-BANCO / ANTI-LIXO)
# ==================================================

BANCO_KEYWORDS = [
    "BANCO", "BB ", "BRADESCO", "CAIXA",
    "SICREDI", "CITIBANK", "DTVM",
    "ASSET", "GESTORA", "GESTÃO"
]

RPPS_PATTERNS = [
    r"\bIPRE[A-Z]{2,}\b",
    r"\bIPREM[A-Z]{2,}\b",
    r"\bIPRES[A-Z]{2,}\b",
    r"\b[A-Z]{3,15}\sPREV\b",
    r"INSTITUTO DE PREVID[ÊE]NCIA[^\n]{0,80}"
]

# ==================================================
# 🔍 INTENÇÃO
# ==================================================

def is_analytical_query(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in [
        "processo", "seleção", "gestor",
        "performance", "satisfação",
        "alocação", "vencimento",
        "macro", "comitê", "presidente"
    ])

def is_summary_query(q: str) -> bool:
    return any(k in q.lower() for k in ["resumo", "resuma", "síntese"])

# ==================================================
# 🔎 BUSCA
# ==================================================

def semantic_search(query: str, k: int = 40):
    vec = np.array([embed(query)]).astype("float32")
    _, idx = INDEX.search(vec, k)
    return [META[i] for i in idx[0]]

# ==================================================
# 🔎 EXTRAÇÕES
# ==================================================

def infer_rpps_from_text(text: str):
    t = text.upper()

    if any(b in t for b in BANCO_KEYWORDS):
        return []

    encontrados = set()
    for p in RPPS_PATTERNS:
        for m in re.findall(p, t):
            nome = m.strip()
            if 6 <= len(nome) <= 60:
                encontrados.add(nome)

    return sorted(encontrados)

def infer_date_from_text(text: str):
    text = text.lower()

    m = re.search(
        r"(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+(20\d{2})",
        text
    )
    if m:
        return f"{m.group(1)} de {m.group(2)}"

    m = re.search(r"(20\d{2})", text)
    if m:
        return m.group(1)

    return "data não identificada"

def temporal_score(meta: dict) -> int:
    ano = meta.get("ano")
    if not ano:
        return 0
    if ano >= CURRENT_YEAR - 1:
        return 3
    if ano >= CURRENT_YEAR - 3:
        return 2
    return 1

# ==================================================
# 🧠 ANSWER
# ==================================================

def answer(query: str) -> str:
    ql = query.lower()

    # --------------------------------------------------
    # 🔹 ANALÍTICO
    # --------------------------------------------------
    if is_analytical_query(ql):

        expansion = """
        comitê de investimentos política de investimentos
        gestores credenciamento seleção avaliação
        renda fixa títulos públicos LTN NTN LFT
        performance rentabilidade meta atuarial
        alocação diretrizes estudo acompanhamento
        """

        docs = semantic_search(expansion + " " + query)

        keywords = [
            "gestor", "gestores", "credenciamento", "seleção",
            "performance", "rentabilidade", "meta atuarial",
            "alocação", "renda fixa", "títulos", "ltn", "ntn", "lft",
            "comitê", "estudo", "avaliação", "acompanhamento"
        ]

        # ----------------------------
        # 1️⃣ FILTRO + SCORE TEMPORAL
        # ----------------------------
        scored_docs = []
        for d in docs:
            text = d.get("text", "")
            if not any(k in text.lower() for k in keywords):
                continue
            scored_docs.append((temporal_score(d), d))

        if not scored_docs:
            scored_docs = [(1, d) for d in docs]

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # ----------------------------
        # 2️⃣ AGRUPA POR RPPS (ANTI-MONOPÓLIO)
        # ----------------------------
        rpps_docs = {}

        for _, d in scored_docs:
            text = d.get("text", "")

            rpps_list = infer_rpps_from_text(text)

            if not rpps_list:
                meta_rpps = d.get("rpps", [])
                if meta_rpps:
                    rpps_list = meta_rpps

            # ⚠️ documentos sem RPPS entram só como CONTEXTO
            if not rpps_list:
                continue

            date = infer_date_from_text(text)

            for rpps in rpps_list:
                rpps_docs.setdefault(rpps, []).append({
                    "date": date,
                    "text": text[:1800]
                })

        if not rpps_docs:
            return (
                "Foram identificados registros institucionais relevantes relacionados "
                "ao tema consultado. No entanto, os documentos não permitem associar "
                "esses registros a RPPS específicos para fins de contato direto, "
                "servindo como indicativos gerais de movimentação institucional."
            )

        # ----------------------------
        # 3️⃣ DIVERSIDADE + ALEATORIEDADE
        # ----------------------------
        rpps_keys = list(rpps_docs.keys())
        random.shuffle(rpps_keys)

        rpps_keys = rpps_keys[:5]  # máx. 5 institutos por resposta

        blocks = []
        for rpps in rpps_keys:
            items = rpps_docs[rpps][:2]  # máx. 2 docs por RPPS
            joined = "\n\n".join(
                f"(Data: {i['date']})\n{i['text']}"
                for i in items
            )
            blocks.append(f"[RPPS: {rpps}]\n{joined}")

        context = "\n\n".join(blocks)

        tone = random.choice([
            "explique de forma analítica",
            "responda como um relatório executivo",
            "responda destacando evidências institucionais",
            "responda de forma técnica e objetiva"
        ])

        prompt = f"""
Você é um analista sênior especializado em RPPS.

Diretrizes:
- Analise o contexto dos documentos.
- Priorize informações mais recentes.
- Identifique sinais institucionais relevantes.
- Não invente nomes, cargos ou números.
- Seja claro, responsável e analítico.

Estilo da resposta: {tone}.

DOCUMENTOS:
{context}

PERGUNTA:
{query}

Responda explicando o que foi identificado para cada RPPS.
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
    # 🔹 RESUMO
    # --------------------------------------------------
    if is_summary_query(ql):
        docs = semantic_search(query, k=12)

        context = "\n\n".join(
            d.get("text", "")[:3000] for d in docs
        )

        prompt = f"""
Você é um analista especializado em RPPS.

Tarefa:
- Produzir um resumo executivo claro
- Destacar decisões, análises e encaminhamentos
- Indicar quando não houver informação explícita

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
    # 🔹 PADRÃO
    # --------------------------------------------------
    docs = semantic_search(query, k=8)

    context = "\n\n".join(
        d.get("text", "")[:2500] for d in docs
    )

    prompt = f"""
Você é um analista especializado em atas de RPPS.

Regras:
- Utilize apenas informações dos documentos
- Não invente dados
- Seja objetivo, mas analítico

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
