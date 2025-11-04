from rag_core.rag_service import build_qa_chain, answer
from rag_core.embeddings.embeddings import get_embedder
from rag_core.llm.ollama_llm import get_llm
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def test_rag_smoke(tmp_path):
    emb = get_embedder("nomic-embed-text", "http://localhost:11434")
    idx = tmp_path / "faiss"

    vs = FAISS.from_documents([Document(page_content="The sky is blue.")], emb)
    vs.save_local(str(idx))
    vs2 = FAISS.load_local(str(idx), emb, allow_dangerous_deserialization=True)

    llm = get_llm("llama3.2:1b", "http://localhost:11434")
    qa = build_qa_chain(vs2, llm, 2)

    ans = answer(qa, "What color is the sky?")
    assert isinstance(ans, str)
