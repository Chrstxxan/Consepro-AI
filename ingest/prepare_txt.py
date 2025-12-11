import os
from pathlib import Path
import unicodedata

from extract_text import extract_text_pdf, extract_text_docx
from ocr_local import ocr_pdf


# ---------------------------------------------------------
# Helpers de limpeza
# ---------------------------------------------------------

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\r", " ").replace("\u00A0", " ")
    t = t.strip()
    return t


def normalize(t: str):
    return "".join(
        c for c in unicodedata.normalize("NFKD", t)
        if not unicodedata.combining(c)
    )


# ---------------------------------------------------------
# Função principal exigida pelo ingest_all.py
# ---------------------------------------------------------

def extract_text_from_document(path):
    """
    Decide automaticamente:
    - Se é PDF → tenta extrair texto normal
        • Se tiver muito pouco texto → faz OCR
    - Se é DOCX → usa extrator de DOCX
    - Se é DOC → ignora ou tenta conversão externa no futuro
    """

    path = Path(path)
    ext = path.suffix.lower()

    # ------------------------------
    # PDF
    # ------------------------------
    if ext == ".pdf":
        try:
            txt = extract_text_pdf(path)
        except Exception:
            txt = ""

        # Se o PDF está praticamente vazio → provavelmente é escaneado
        if not txt or len(txt.strip()) < 30:
            print(f"[OCR LOCAL] Documento escaneado detectado → {path}")
            try:
                txt = ocr_pdf(path)
            except Exception as e:
                print(f"[OCR LOCAL] Erro ao fazer OCR em {path}: {e}")
                txt = ""

        return clean_text(txt)

    # ------------------------------
    # DOCX
    # ------------------------------
    elif ext == ".docx":
        try:
            txt = extract_text_docx(path)
            return clean_text(txt)
        except Exception as e:
            print(f"[DOCX] Erro ao extrair texto de {path}: {e}")
            return ""

    # ------------------------------
    # DOC (não suportado)
    # ------------------------------
    elif ext == ".doc":
        print(f"[WARN] Arquivo .doc não suportado diretamente: {path}")
        return ""

    # ------------------------------
    # Outro tipo inesperado
    # ------------------------------
    else:
        print(f"[WARN] Tipo de arquivo desconhecido: {path}")
        return ""
