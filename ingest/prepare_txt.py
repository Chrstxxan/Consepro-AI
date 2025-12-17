import os
import re
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

RAW_TXT_ROOT = "data/raw_txt"          # onde chegam os txt extraídos do PDF
OUT_ROOT = "data/processed_txt"        # pasta final
INVEST_DIR = os.path.join(OUT_ROOT, "investimentos")
ADMIN_DIR = os.path.join(OUT_ROOT, "administrativos")

os.makedirs(INVEST_DIR, exist_ok=True)
os.makedirs(ADMIN_DIR, exist_ok=True)

# --------------------------------------------------
# PALAVRAS-CHAVE DE INVESTIMENTOS
# --------------------------------------------------

KEYWORDS_INVEST = [
    "comitê de investimentos",
    "politica de investimentos",
    "política de investimentos",
    "alocação",
    "renda fixa",
    "renda variável",
    "gestor",
    "gestores",
    "benchmark",
    "meta atuarial",
    "rentabilidade",
    "performance",
    "aplicações",
    "investimentos",
    "fundo",
    "títulos públicos",
    "títulos federais",
]

# --------------------------------------------------
# UTILIDADES
# --------------------------------------------------

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text

def is_investment_doc(text: str) -> bool:
    text = normalize(text)
    hits = sum(1 for k in KEYWORDS_INVEST if k in text)
    return hits >= 2   # regra segura

def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 3:
            continue
        lines.append(line)
    return "\n".join(lines)

# --------------------------------------------------
# PROCESSAMENTO
# --------------------------------------------------

def process_all():
    txt_files = list(Path(RAW_TXT_ROOT).rglob("*.txt"))

    print(f"📄 TXT encontrados: {len(txt_files)}")

    invest_count = 0
    admin_count = 0

    for path in txt_files:
        try:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            text = clean_text(raw_text)

            if not text:
                continue

            if is_investment_doc(text):
                out_dir = INVEST_DIR
                invest_count += 1
            else:
                out_dir = ADMIN_DIR
                admin_count += 1

            out_path = os.path.join(out_dir, path.name)
            Path(out_path).write_text(text, encoding="utf-8")

        except Exception as e:
            print(f"[ERRO] {path}: {e}")

    print("✅ Processamento concluído")
    print(f"📊 Investimentos: {invest_count}")
    print(f"📊 Administrativos: {admin_count}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    process_all()
