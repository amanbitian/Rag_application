import streamlit as st
import logging, traceback

from rag_core.logging_conf import setup_logging
from rag_core.config import load_settings
from rag_core.embeddings.embeddings import get_embedder
from rag_core.llm.ollama_llm import get_llm
from rag_core.loaders.pdf_loader import load_pdf
from rag_core.loaders.git_loader import load_repo
from rag_core.vectorstores.faiss_store import build_or_update_faiss
from langchain_community.vectorstores import FAISS

setup_logging()
logger = logging.getLogger("ui")

st.set_page_config(page_title="RAG Corp (Robust)", layout="wide")
st.title("🏢 RAG Corp — Robust PDF & GitHub RAG")

try:
    cfg = load_settings()
except Exception as e:
    st.error(f"Failed to load settings.yaml: {e}")
    st.stop()

with st.sidebar:
    st.subheader("Models")
    embed_pref = st.text_input("Embedding (preferred via Ollama)", cfg.embed_model)
    llm_model = st.text_input("LLM (Ollama)", cfg.llm_model)
    st.caption("If Ollama embeddings fail, we auto-fallback to HF (all-MiniLM-L6-v2).")

mode = st.radio("Source Type", ["📄 PDF", "💻 GitHub Repo"], horizontal=True)
docs = None

try:
    if mode == "📄 PDF":
        file = st.file_uploader("Upload a PDF", type=["pdf"])
        if st.button("Index PDF") and file:
            docs = load_pdf(file.read())

    elif mode == "💻 GitHub Repo":
        url = st.text_input("GitHub repo URL (https://github.com/user/repo)")
        if st.button("Index Repo") and url:
            docs = load_repo(url, cfg.git_branch, cfg.git_include_exts, cfg.git_exclude_dirs)

    if docs:
        st.info(f"Loaded {len(docs)} documents. Building/updating FAISS index…")
        embedder = get_embedder(embed_pref, cfg.ollama_base_url)
        vs = build_or_update_faiss(docs, embedder, cfg.index_dir, cfg.chunk_size, cfg.chunk_overlap)
        st.session_state["index_ready"] = True
        st.success("✅ Index ready")

except Exception as e:
    logger.exception("Indexing failure.")
    st.error(f"Indexing failed: {e}")
    with st.expander("Details"):
        st.code("".join(traceback.format_exc()))
    st.stop()

if st.session_state.get("index_ready"):
    q = st.text_input("Ask a question")
    if q:
        try:
            embedder = get_embedder(embed_pref, cfg.ollama_base_url)
            vs = FAISS.load_local(cfg.index_dir, embedder, allow_dangerous_deserialization=True)
            llm = get_llm(llm_model, cfg.ollama_base_url)
            from rag_core.rag_service import build_qa_chain, answer
            qa = build_qa_chain(vs, llm, cfg.retriever_k)
            with st.spinner("Thinking…"):
                ans = answer(qa, q)
            st.markdown("### 🧠 Answer")
            st.write(ans)
        except Exception as e:
            logger.exception("QA failure.")
            st.error(f"QA failed: {e}")
            with st.expander("Details"):
                st.code("".join(traceback.format_exc()))
