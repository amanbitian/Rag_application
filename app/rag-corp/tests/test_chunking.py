from rag_core.vectorstores.faiss_store import build_or_update_faiss
from rag_core.embeddings.embeddings import get_embedder
from langchain.schema import Document

def test_chunking_and_index(tmp_path):
    docs = [Document(page_content="hello " * 300)]
    idx = tmp_path / "faiss"
    emb = get_embedder("nomic-embed-text", "http://localhost:11434")

    vs = build_or_update_faiss(docs, emb, str(idx), 100, 20)
    assert vs is not None
