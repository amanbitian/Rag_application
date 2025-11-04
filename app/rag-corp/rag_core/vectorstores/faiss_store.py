import logging, os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def build_or_update_faiss(docs, embedder, index_dir: str, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No text chunks found. PDF might be scanned or empty.")

    # Probe once to ensure embedder returns non-empty vectors
    probe = embedder.embed_query(chunks[0].page_content[:200] or "test")
    if not probe:
        raise RuntimeError("Embedding backend returned empty vector. Check configuration.")

    if os.path.isdir(index_dir) and os.listdir(index_dir):
        logger.info("Loading existing FAISS index: %s", index_dir)
        vs = FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)
        vs.add_documents(chunks)
        vs.save_local(index_dir)
        logger.info("Updated FAISS index with %d chunks.", len(chunks))
        return vs
    else:
        os.makedirs(index_dir, exist_ok=True)
        logger.info("Building new FAISS index at: %s", index_dir)
        vs = FAISS.from_documents(chunks, embedder)
        vs.save_local(index_dir)
        logger.info("Built FAISS index with %d chunks.", len(chunks))
        return vs
