from langchain_community.document_loaders import PyPDFLoader
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

def load_pdf(file_bytes: bytes):
    """
    Load a PDF from raw bytes, using OCR fallback if no text found.
    """
    docs = []
    try:
        loader = PyPDFLoader(io.BytesIO(file_bytes))
        docs = loader.load()
    except Exception as e:
        logger.warning(f"PyPDFLoader failed, will try OCR: {e}")

    # if loader returns no text
    if not docs or all(len(d.page_content.strip()) == 0 for d in docs):
        logger.info("⚠️ No text found — attempting OCR fallback...")
        docs = ocr_pdf(file_bytes)

    if not docs:
        raise ValueError("No text chunks found. PDF might be scanned or empty.")
    return docs


def ocr_pdf(file_bytes: bytes):
    """Convert each page image to text using Tesseract OCR."""
    texts = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    for i, page in enumerate(pdf):
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        texts.append({
            "page_content": text,
            "metadata": {"source": f"page_{i+1}", "page_number": i+1}
        })
    pdf.close()
    return [type("Doc", (), d) for d in texts]  # mimic LC docs
