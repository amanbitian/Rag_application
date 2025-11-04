import tempfile, os, logging
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

def load_pdf(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        docs = PyPDFLoader(path).load()
        logger.info("Loaded PDF with %d pages.", len(docs))
        return docs
    finally:
        os.remove(path)
