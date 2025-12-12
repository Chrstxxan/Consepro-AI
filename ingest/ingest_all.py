import os
from tqdm import tqdm
from extract_text import extract_any_text
from ocr_local import ocr_local
from prepare_txt import clean_text
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

ROOT = "data"                   # <--- pasta do scraper
OUT = os.path.join(ROOT, "processed_txt")
os.makedirs(OUT, exist_ok=True)

VALID_EXTS = {".pdf", ".doc", ".docx"}

def list_all_documents():
    for root, dirs, files in os.walk(ROOT):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_EXTS:
                yield os.path.join(root, f)

def output_path(original_path):
    """Gera caminho paralelo em processed_txt/ preservando estrutura."""
    rel = os.path.relpath(original_path, ROOT)
    out = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out.replace(".pdf",".txt").replace(".docx",".txt").replace(".doc",".txt")

from pdf_state import load_state, save_state, sha256_file
from prepare_txt import extract_text_from_document  # usa OCR + extrator normal

state = load_state()

def ingest_one(pdf_path: Path):
    pdf_path = Path(pdf_path)
    pdf_key = str(pdf_path)

    pdf_hash = sha256_file(pdf_path)

    # Se já existia e hash não mudou → pular
    if pdf_key in state and state[pdf_key]["sha256"] == pdf_hash:
        print(f"[SKIP] {pdf_path} (sem mudanças)")
        return state[pdf_key]["processed_txt"]

    # Processa (OCR ou texto normal)
    txt = extract_text_from_document(pdf_path)

    # Caminho correto dentro de data/processed_txt/
    out_path = Path(output_path(str(pdf_path)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt, encoding="utf-8")

    # Atualiza estado
    state[pdf_key] = {
        "sha256": pdf_hash,
        "processed_txt": str(out_path)
    }
    save_state(state)

    print(f"[OK] Processado {pdf_path}")
    return str(out_path)

def run():
    docs = list(list_all_documents())
    print(f"📄 Total de documentos para ingestão: {len(docs)}")

    for p in tqdm(docs):
        ingest_one(p)

    print("🎉 Ingestão concluída com sucesso!")

if __name__ == "__main__":
    run()
