import json
import re
from pathlib import Path

META_PATH = Path("embeddings/metadata.json")

BANCO_BLACKLIST = [
    "BANCO", "BB ", "BRADESCO", "CAIXA",
    "SICREDI", "CITIBANK", "DTVM", "ASSET",
    "GESTÃO", "GESTORA"
]

RPPS_INVALIDOS = [
    "INVESTIMENTOS DO RPPS",
    "NÍVEL BÁSICO",
    "CP RPPS",
    "CGINV",
    "ONLINE",
    "CURSO",
    "CERTIFICAÇÃO",
    "ADOR",
    "INSTITUTO DE PREV"
]

RPPS_PATTERNS = [
    r"\bIPRE[A-Z]{2,}\b",
    r"\bIPRES[A-Z]{2,}\b",
    r"\bIPREM[A-Z]{2,}\b",
    r"\b[A-Z\s]{3,40} PREV\b",
    r"INSTITUTO DE PREVID[ÊE]NCIA[^\n]{0,80}"
]

def is_banco(text: str) -> bool:
    t = text.upper()
    return any(b in t for b in BANCO_BLACKLIST)

def rpps_valido(nome: str) -> bool:
    n = nome.upper().strip()

    if len(n) < 10:
        return False

    if any(x in n for x in RPPS_INVALIDOS):
        return False

    return any(x in n for x in ["IPRE", "PREV", "INSTITUTO DE PREVIDÊNCIA"])

def normalizar(nome: str) -> str:
    nome = nome.replace("\n", " ")
    nome = re.sub(r"\s{2,}", " ", nome)
    return nome.title().strip()

def extract_rpps_from_text(text: str):
    t = text.upper()
    encontrados = set()

    if is_banco(t):
        return []

    for p in RPPS_PATTERNS:
        for m in re.findall(p, t):
            nome = normalizar(m)
            if rpps_valido(nome):
                encontrados.add(nome)

    return sorted(encontrados)

def main():
    data = json.loads(META_PATH.read_text(encoding="utf-8"))

    preenchidos = 0
    limpos = 0

    for d in data:
        texto = d.get("text", "")

        # 1️⃣ Limpa RPPS existentes (isso é o pulo do gato)
        rpps_limpos = []
        for r in d.get("rpps", []):
            r2 = normalizar(r)
            if rpps_valido(r2):
                rpps_limpos.append(r2)

        # 2️⃣ Se não sobrou nada, tenta extrair de novo
        if not rpps_limpos and texto:
            rpps_limpos = extract_rpps_from_text(texto)
            if rpps_limpos:
                preenchidos += 1

        if rpps_limpos:
            limpos += 1

        d["rpps"] = rpps_limpos
        d["rpps_canonico"] = rpps_limpos[0] if rpps_limpos else None

    backup = META_PATH.with_suffix(".backup.json")
    backup.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    META_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("✅ Metadata lapidada com sucesso")
    print(f"🧹 RPPS limpos/normalizados: {limpos}")
    print(f"🔧 RPPS preenchidos via texto: {preenchidos}")
    print(f"🗂️ Backup criado em: {backup}")

if __name__ == "__main__":
    main()
