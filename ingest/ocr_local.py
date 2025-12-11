import fitz
import pytesseract
from PIL import Image

# Caminho fixo do Tesseract no Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_local(pdf_path):
    """
    Executa OCR offline página por página usando Tesseract.
    """
    try:
        doc = fitz.open(pdf_path)
    except:
        return ""

    pages = []

    for i, page in enumerate(doc):
        try:
            pix = page.get_pixmap()
            mode = "RGB" if pix.alpha == 0 else "RGBA"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

            text = pytesseract.image_to_string(img, lang="por")
            pages.append(text)
        except Exception as e:
            print(f"[OCR LOCAL] Erro na página {i} do PDF {pdf_path}: {e}")
            pages.append("")

    return "\n".join(pages).strip()


#  Wrapper necessário para compatibilidade com prepare_txt.py
def ocr_pdf(path):
    """
    Wrapper para manter compatibilidade com o pipeline.
    Apenas chama ocr_local().
    """
    return ocr_local(path)
