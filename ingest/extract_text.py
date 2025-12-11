import os
import fitz  # PyMuPDF
import docx
import subprocess
from dotenv import load_dotenv
load_dotenv()

def extract_pdf_text(path):
    """Extrai texto de PDFs. Se falhar, retorna ''. """
    try:
        doc = fitz.open(path)
        txt = []
        for pg in doc:
            txt.append(pg.get_text("text") or "")
        doc.close()
        return "\n".join(txt).strip()
    except:
        return ""

def extract_docx_text(path):
    """Extrai texto de documentos docx."""
    try:
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    except:
        return ""

def extract_doc_text(path):
    """Extrai texto de arquivos .doc via antiword (se existir)."""
    try:
        result = subprocess.run(["antiword", path], capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return ""

def extract_any_text(path: str) -> str:
    """Escolhe automaticamente o método correto."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return extract_pdf_text(path)

    if ext == ".docx":
        return extract_docx_text(path)

    if ext == ".doc":
        return extract_doc_text(path)

    return ""

def extract_text_pdf(path):
    return extract_pdf_text(str(path))

def extract_text_docx(path):
    return extract_docx_text(str(path))

