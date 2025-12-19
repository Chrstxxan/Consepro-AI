import os
import json
import faiss
import numpy as np
import re
import random
import openai
from datetime import datetime
from embeddings.embedder import embed

# 🆕 fuzzy matching
from rapidfuzz import fuzz

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
    r"\bIPREV[A-Z]{2,}\b",
    r"\b[A-Z]{3,15}\sPREV\b",
    r"INSTITUTO DE PREVID[ÊE]NCIA[^\n]{0,80}"
]

# ==================================================
# 🆕 NORMALIZAÇÃO CANÔNICA
# ==================================================

def normalize_rpps_name(name: str) -> str:
    if not name:
        return ""

    n = name.upper()
    n = re.sub(r"[^A-Z ]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()

    blacklist = [
        "INSTITUTO DE PREVIDENCIA",
        "INSTITUTO DE PREVIDÊNCIA",
        "INSTITUTO PREVIDENCIA",
        "DOS SERVIDORES PUBLICOS",
        "DOS SERVIDORES PÚBLICOS",
        "SERVIDORES PUBLICOS",
        "SERVIDORES PÚBLICOS",
        "MUNICIPIO DE",
        "MUNICÍPIO DE"
    ]

    for b in blacklist:
        n = n.replace(b, "")

    return n.strip()

# ==================================================
# 🆕 SIGLA ÂNCORA
# ==================================================

def extract_sigla(name: str):
    m = re.search(
        r"\bIPRE[A-Z]{2,6}\b|\bIPRES[A-Z]{2,6}\b|\bIPREV[A-Z]{2,6}\b",
        name
    )
    return m.group(0) if m else None

# ==================================================
# 🆕 MATCH DE RPPS (SIGLA + FUZZY)
# ==================================================

def is_same_rpps(a: str, b: str) -> bool:
    a_norm = normalize_rpps_name(a)
    b_norm = normalize_rpps_name(b)

    if not a_norm or not b_norm:
        return False

    # 1️⃣ match direto
    if a_norm == b_norm:
        return True

    # 2️⃣ sigla âncora
    sig_a = extract_sigla(a_norm)
    sig_b = extract_sigla(b_norm)
    if sig_a and sig_b and sig_a == sig_b:
        return True

    # 3️⃣ fuzzy (fallback)
    score = fuzz.token_set_ratio(a_norm, b_norm)
    return score >= 85

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
            nome = normalize_rpps_name(m.strip())
            if 6 <= len(nome) <= 60:
                encontrados.add(nome)

    return sorted(encontrados)

def infer_date_from_text(text: str):
    text = text.lower()
    m = re.search(r"(20\d{2})", text)
    return m.group(1) if m else "data não identificada"

# ==================================================
# 🆕 TOP N ATAS POR RPPS (INALTERADO)
# ==================================================

def get_top_docs_for_rpps(rpps_name, keywords, limit):
    docs = []

    for d in META:
        meta_rpps = [
            normalize_rpps_name(r)
            for r in d.get("rpps", [])
        ]

        # 🆕 match inteligente
        if not any(is_same_rpps(rpps_name, mr) for mr in meta_rpps):
            continue

        ano = d.get("ano")
        if not ano:
            continue

        text = d.get("text", "").lower()
        if not any(k in text for k in keywords):
            continue

        docs.append(d)

    docs.sort(key=lambda x: x.get("ano", 0), reverse=True)
    return docs[:limit]

# ==================================================
# 🧠 ANSWER
# ==================================================

def answer(query: str) -> str:
    ql = query.lower()

    if is_analytical_query(ql):

        expansion = """
        comitê de investimentos política de investimentos
        gestores credenciamento seleção avaliação
        renda fixa títulos públicos LTN NTN LFT
        performance rentabilidade meta atuarial
        alocação diretrizes estudo acompanhamento
        """

        _ = semantic_search(expansion + " " + query)

        keywords = [
            "gestor", "gestores", "credenciamento", "seleção",
            "performance", "rentabilidade", "meta atuarial",
            "alocação", "renda fixa", "títulos", "ltn", "ntn", "lft",
            "comitê", "estudo", "avaliação", "acompanhamento"
        ]

        target_rpps = infer_rpps_from_text(query)
        blocks = []

        # --------------------------------------------------
        # 🔹 RPPS ESPECÍFICO → TOP 8
        # --------------------------------------------------
        if target_rpps:
            rpps = target_rpps[0]
            docs = get_top_docs_for_rpps(rpps, keywords, limit=8)

            for d in docs:
                blocks.append(
                    f"[RPPS: {rpps}]\n(Ano: {d.get('ano')})\n{d.get('text','')[:1800]}"
                )

        # --------------------------------------------------
        # 🔹 PERGUNTA ABERTA → TOP 5 POR RPPS
        # --------------------------------------------------
        else:
            all_rpps = set()
            for m in META:
                for r in m.get("rpps", []):
                    all_rpps.add(normalize_rpps_name(r))

            all_rpps = list(all_rpps)
            random.shuffle(all_rpps)

            for rpps in all_rpps:
                docs = get_top_docs_for_rpps(rpps, keywords, limit=5)
                if not docs:
                    continue

                joined = "\n\n".join(
                    f"(Ano: {d.get('ano')})\n{d.get('text','')[:1200]}"
                    for d in docs
                )

                blocks.append(f"[RPPS: {rpps}]\n{joined}")

                if len(blocks) >= 20:
                    break

        if not blocks:
            return (
                "Os documentos analisados não apresentam informações "
                "suficientes e recentes relacionadas à pergunta."
            )

        context = "\n\n".join(blocks)

        prompt = f"""
Você é um analista sênior especializado em RPPS.

Diretrizes:
- Utilize exclusivamente os documentos fornecidos.
- Não invente nomes, cargos ou números.
- Agrupe informações por RPPS.

DOCUMENTOS:
{context}

PERGUNTA:
{query}
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
    # 🔹 OUTROS MODOS (INALTERADOS)
    # --------------------------------------------------

    docs = semantic_search(query, k=8)
    context = "\n\n".join(d.get("text", "")[:2500] for d in docs)

    prompt = f"""
Você é um analista especializado em atas de RPPS.

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
