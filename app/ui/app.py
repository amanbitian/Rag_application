import streamlit as st
import logging, traceback
import time
from rag_core.llm.ollama_utils import list_ollama_models
from rag_core.llm.ollama_metrics import (generate_with_stats, get_model_info)
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
    show_ctx = st.toggle("Show retrieved context under answers", value=False)

    if q:
        try:
            # Reuse the same embedder used for indexing (or recreate)
            embedder = get_embedder(embed_pref, cfg.ollama_base_url)
            vs = FAISS.load_local(cfg.index_dir, embedder, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Failed to load vector index: {e}")
            st.stop()

        if not llm_choices:
            st.warning("Select at least one LLM from the sidebar.")
            st.stop()

        # 🔎 Retrieve once, use same context for all models
        k = cfg.retriever_k
        retrieved = vs.similarity_search(q, k=k)
        context = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(retrieved)])

        # Simple prompt template (you can customize)
        prompt = (
            "You are a helpful assistant. Use ONLY the context to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\n"
            "Answer:"
        )

        st.markdown("### 🧪 Model Comparison")

        # 🧩 Side-by-side layout
        cols = st.columns(len(llm_choices))
        results = []

        # Prepare common Ollama options
        options = {"temperature": temp, "seed": int(seed)}

        for i, model_name in enumerate(llm_choices):
            with cols[i]:
                try:
                    # Call Ollama directly to capture metrics
                    text, stats = generate_with_stats(
                        base_url=cfg.ollama_base_url,
                        model=model_name,
                        prompt=prompt,
                        options=options,
                    )
                    card = get_model_info(cfg.ollama_base_url, model_name)
                    results.append({"model": model_name, "text": text, "stats": stats, "card": card, "error": None})
                except Exception as e:
                    results.append({"model": model_name, "text": "", "stats": {}, "card": {}, "error": str(e)})

        # Render result cards (still side-by-side using the same 'cols' again)
        cols = st.columns(len(llm_choices))
        for i, r in enumerate(results):
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"#### 🧠 {r['model']}")
                    if r["error"]:
                        st.error(r["error"])
                        continue

                    # 🏷️ High-level metrics
                    s = r["stats"]
                    st.metric("Latency (wall)", f"{s.get('latency_wall_s', 0):.2f}s")
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.caption(f"Prompt tokens: {s.get('prompt_eval_count', 0)}")
                        st.caption(f"Load time: {s.get('load_duration_s', 0):.2f}s")
                    with mcol2:
                        st.caption(f"Gen tokens: {s.get('eval_count', 0)}")
                        st.caption(f"Tokens/sec: {s.get('tokens_per_s', 0):.1f}")

                    # Model card tidbits (if available)
                    card = r["card"] or {}
                    if card:
                        st.caption(
                            "Ctx window: "
                            f"{card.get('context_length', '—')} • "
                            f"Params: {card.get('parameter_size', '—')} • "
                            f"Quant: {card.get('quantization_level', '—')}"
                        )

                    # The answer
                    st.markdown("---")
                    st.write(r["text"])

                    if show_ctx:
                        with st.expander("Show retrieved context"):
                            for j, d in enumerate(retrieved, start=1):
                                st.markdown(f"**[{j}]** {d.metadata.get('source', '')}")
                                st.write(d.page_content[:1200] + ("..." if len(d.page_content) > 1200 else ""))


