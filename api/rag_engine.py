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
    m = re.search(r"(20\d{2})", text)
    return m.group(1) if m else "data não identificada"

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
# 🆕 ADIÇÃO GLOBAL — LISTA REAL DE RPPS
# ==================================================

def get_all_rpps():
    rpps = set()
    for m in META:
        for r in m.get("rpps", []):
            rpps.add(r)
    return sorted(rpps)

# ==================================================
# 🆕 ADIÇÃO GLOBAL — 1 DOC RECENTE POR RPPS
# ==================================================

def get_recent_doc_for_rpps(rpps, keywords):
    candidates = []

    for d in META:
        if rpps not in d.get("rpps", []):
            continue

        ano = d.get("ano")
        if not ano or ano < CURRENT_YEAR - 3:
            continue

        text = d.get("text", "").lower()
        if not any(k in text for k in keywords):
            continue

        candidates.append(d)

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: x.get("ano", 0), reverse=True)[0]

# ==================================================
# 🧠 ANSWER
# ==================================================

def answer(query: str) -> str:
    ql = query.lower()

    # --------------------------------------------------
    # 🔹 MODO ANALÍTICO
    # --------------------------------------------------
    if is_analytical_query(ql):

        expansion = """
        comitê de investimentos política de investimentos
        gestores credenciamento seleção avaliação
        renda fixa títulos públicos LTN NTN LFT
        performance rentabilidade meta atuarial
        alocação diretrizes estudo acompanhamento
        """

        # mantém FAISS (não removido)
        _ = semantic_search(expansion + " " + query)

        keywords = [
            "gestor", "gestores", "credenciamento", "seleção",
            "performance", "rentabilidade", "meta atuarial",
            "alocação", "renda fixa", "títulos", "ltn", "ntn", "lft",
            "comitê", "estudo", "avaliação", "acompanhamento"
        ]

        # ==================================================
        # 🆕 ADIÇÃO CRÍTICA — ENTIDADE FIRST (PERGUNTA ABERTA)
        # ==================================================

        all_rpps = get_all_rpps()
        random.shuffle(all_rpps)

        blocks = []
        MAX_RPPS_TOTAL = 20

        for rpps in all_rpps:
            doc = get_recent_doc_for_rpps(rpps, keywords)
            if not doc:
                continue

            text = doc.get("text", "")[:1500]
            ano = doc.get("ano")

            blocks.append(
                f"[RPPS: {rpps}]\n(Ano: {ano})\n{text}"
            )

            if len(blocks) >= MAX_RPPS_TOTAL:
                break

        if not blocks:
            return (
                "Foram analisados diversos RPPS, porém não foram encontrados "
                "registros recentes e relevantes relacionados ao tema consultado."
            )

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
- Pergunta aberta: listar o MAIOR NÚMERO POSSÍVEL de RPPS.
- Cada RPPS representa uma entidade distinta.
- Priorize documentos recentes.
- Não invente nomes, cargos ou números.
- Seja direto e orientado à decisão.

Estilo da resposta: {tone}.

DOCUMENTOS:
{context}

PERGUNTA:
{query}

Responda analisando cada RPPS listado.
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
    # 🔹 MODO RESUMO (INALTERADO)
    # --------------------------------------------------
    if is_summary_query(ql):
        docs = semantic_search(query, k=12)
        context = "\n\n".join(d.get("text", "")[:3000] for d in docs)

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
    # 🔹 BUSCA PADRÃO (INALTERADO)
    # --------------------------------------------------
    docs = semantic_search(query, k=8)
    context = "\n\n".join(d.get("text", "")[:2500] for d in docs)

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
