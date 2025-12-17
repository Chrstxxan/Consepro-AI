import os
from pathlib import Path
from extract_text import extract_any_text
from ocr_local import ocr_pdf
from pdf_state import load_state, save_state, sha256_file

PDF_DIR = "data"
RAW_TXT_DIR = "data/raw_txt"

os.makedirs(RAW_TXT_DIR, exist_ok=True)

state = load_state()

def process_document(path: Path):
    try:
        file_hash = sha256_file(path)

        if state.get(str(path)) == file_hash:
            return

        text = extract_any_text(str(path))

        if not text or len(text.strip()) < 50:
            text = ocr_pdf(str(path))

        if not text or len(text.strip()) < 50:
            print(f"[SKIP] Sem texto útil: {path}")
            return

        out = Path(RAW_TXT_DIR) / (path.stem + ".txt")
        out.write_text(text, encoding="utf-8")

        state[str(path)] = file_hash
        print(f"✅ Processado: {path}")

    except Exception as e:
        print(f"[ERRO] {path}: {e}")

def main():
    pdfs = [
        p for p in Path(PDF_DIR).rglob("*.*")
        if p.suffix.lower() in [".pdf", ".doc", ".docx"]
        and "raw_txt" not in p.parts
        and "processed_txt" not in p.parts
    ]

    print(f"📄 Documentos encontrados: {len(pdfs)}")

    for doc in pdfs:
        process_document(doc)

    save_state(state)
    print("🎉 Ingest finalizado")

if __name__ == "__main__":
    main()
