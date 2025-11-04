import streamlit as st
import logging, traceback
import time
from rag_core.llm.ollama_utils import list_ollama_models
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

    # Discover models from Ollama (cached)
    @st.cache_data(ttl=60)
    def _models(base):
        return list_ollama_models(base)

    available = _models(cfg.ollama_base_url)

    embed_pref = st.selectbox(
        "Embedding model (preferred via Ollama; auto-fallback to HF if needed)",
        options=["bge-m3", "nomic-embed-text", "all-MiniLM-L6-v2"], index=0
    )

    llm_choices = st.multiselect(
        "LLMs for comparison (select 1–4)",
        options=available,
        default=[m for m in available if m.startswith("llama3")][:1] or available[:1],
        max_selections=4
    )

    temp = st.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
    seed = st.number_input("Seed", value=42, step=1)

    st.caption("Tip: Keep 1–4 models for a fair, fast comparison.")

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

# Q&A multi-model comparison
if st.session_state.get("index_ready"):
    q = st.text_input("Ask a question", key="qq")
    if q:
        try:
            embedder = get_embedder(embed_pref, cfg.ollama_base_url)
            vs = FAISS.load_local(cfg.index_dir, embedder, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Failed to load vector index: {e}")
            st.stop()

        if not llm_choices:
            st.warning("Select at least one LLM from the sidebar.")
            st.stop()

        st.markdown("### 🧪 Model Comparison")
        cols = st.columns(len(llm_choices))
        results = []

        from rag_core.rag_service import build_qa_chain, answer

        for i, model_name in enumerate(llm_choices):
            with cols[i]:
                try:
                    llm = get_llm(model_name, cfg.ollama_base_url) #, temperature=temp, seed=seed)
                    qa = build_qa_chain(vs, llm, cfg.retriever_k)
                    t0 = time.perf_counter()
                    ans = answer(qa, q)
                    dt = time.perf_counter() - t0
                    results.append({"model": model_name, "answer": ans, "latency": dt, "error": None})
                except Exception as e:
                    results.append({"model": model_name, "answer": "", "latency": None, "error": str(e)})

        for r in results:
            with st.container(border=True):
                st.markdown(f"#### 🧠 {r['model']}")
                if r["error"]:
                    st.error(f"Error: {r['error']}")
                    continue
                st.markdown(f"**Latency:** {r['latency']:.2f}s")
                st.write(r["answer"])

