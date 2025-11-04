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
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from rag_core.telemetry.sheets_logger import log_rows
from rag_core.llm.ollama_metrics import generate_with_stats, get_model_info
from rag_core.llm.ollama_utils import list_ollama_models
import uuid, os
# print("🔐 GSheets creds path:", os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
# print("📄 Sheet ID:", os.getenv("SHEET_ID"))

setup_logging()
logger = logging.getLogger("ui")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]  # short session id

st.set_page_config(page_title="RAG Corp (Robust)", layout="wide")
st.title("🏢 RAG Corp — Robust PDF & GitHub RAG")

try:
    cfg = load_settings()
except Exception as e:
    st.error(f"Failed to load settings.yaml: {e}")
    st.stop()

with st.sidebar:
    st.subheader("Models")

    @st.cache_data(ttl=60)
    def _models(base):
        return list_ollama_models(base)

    available = _models(cfg.ollama_base_url)

    embed_pref = st.selectbox(
        "Embedding model (preferred via Ollama; fallback → HF)",
        options=["bge-m3", "nomic-embed-text", "all-MiniLM-L6-v2"],
        index=0
    )

    llm_choices = st.multiselect(
        "LLMs for comparison (select 1–4)",
        options=available,
        default=[m for m in available if m.startswith("llama3")] or available[:1],
        max_selections=4
    )

    temp = st.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
    seed = st.number_input("Seed", value=42, step=1)

    concurrent_run = st.toggle("Run models concurrently (multi-thread)", value=True, help="Turn off to run sequentially")


    st.caption("Tip: Keep 1–4 models for a fair, fast comparison.")

mode = st.radio("Source Type", ["📄 PDF", "💻 GitHub Repo"], horizontal=True)
docs = None

try:
    if mode == "📄 PDF":
        file = st.file_uploader("Upload a PDF", type=["pdf"])
        if st.button("Index PDF") and file:
            # keep for indexing
            docs = load_pdf(file.read())

            #  remember the uploaded file name(s) for logging
            st.session_state.uploaded_files = [file]
            st.session_state.last_indexed_files = [file.name]
            st.session_state.pop("last_indexed_repo", None)  # clear repo marker



    elif mode == "💻 GitHub Repo":

        url = st.text_input("GitHub repo URL (https://github.com/user/repo)")

        if st.button("Index Repo") and url:
            docs = load_repo(url, cfg.git_branch, cfg.git_include_exts, cfg.git_exclude_dirs)

            # remember the repo for logging
            st.session_state.last_indexed_repo = url
            st.session_state.pop("uploaded_files", None)  # clear file markers
            st.session_state.pop("last_indexed_files", None)

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
        # Load index
        try:
            embedder = get_embedder(embed_pref, cfg.ollama_base_url)
            vs = FAISS.load_local(cfg.index_dir, embedder, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Failed to load vector index: {e}")
            st.stop()

        if not llm_choices:
            st.warning("Select at least one LLM from the sidebar.")
            st.stop()

        # Retrieve once for fairness
        k = cfg.retriever_k
        retrieved = vs.similarity_search(q, k=k)
        context = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(retrieved)])

        prompt = (
            "You are a helpful assistant. Use ONLY the context to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\n"
            "Answer:"
        )

        st.markdown("### 🧪 Model Comparison")

        options = {"temperature": float(temp), "seed": int(seed)}
        results = []

        def _run_model(mname: str):
            # Call Ollama directly to get metrics
            text, stats = generate_with_stats(
                base_url=cfg.ollama_base_url,
                model=mname,
                prompt=prompt,
                options=options,
            )
            card = get_model_info(cfg.ollama_base_url, mname)
            return {"model": mname, "text": text, "stats": stats, "card": card, "error": None}

        # --- concurrency toggle ---
        if concurrent_run and len(llm_choices) > 1:
            with ThreadPoolExecutor(max_workers=len(llm_choices)) as ex:
                futs = {ex.submit(_run_model, m): m for m in llm_choices}
                for f in as_completed(futs):
                    try:
                        results.append(f.result())
                    except Exception as e:
                        results.append({"model": futs[f], "text": "", "stats": {}, "card": {}, "error": str(e)})
        else:
            for m in llm_choices:
                try:
                    results.append(_run_model(m))
                except Exception as e:
                    results.append({"model": m, "text": "", "stats": {}, "card": {}, "error": str(e)})

        # side-by-side render
        cols = st.columns(len(llm_choices))
        for i, r in enumerate(results):
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"#### 🧠 {r['model']}")
                    if r["error"]:
                        st.error(r["error"])
                        continue

                    s = r["stats"]
                    st.metric("Latency (wall)", f"{s.get('latency_wall_s', 0):.2f}s")
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.caption(f"Prompt tokens: {s.get('prompt_eval_count', 0)}")
                        st.caption(f"Load time: {s.get('load_duration_s', 0):.2f}s")
                    with mcol2:
                        st.caption(f"Gen tokens: {s.get('eval_count', 0)}")
                        st.caption(f"Tokens/sec: {s.get('tokens_per_s', 0):.1f}")

                    card = r["card"] or {}
                    st.caption(
                        f"Ctx: {card.get('context_length','—')} • "
                        f"Params: {card.get('parameter_size','—')} • "
                        f"Quant: {card.get('quantization_level','—')}"
                    )

                    st.markdown("---")
                    st.write(r["text"])

                    if show_ctx:
                        with st.expander("Show retrieved context"):
                            for j, d in enumerate(retrieved, start=1):
                                st.markdown(f"**[{j}]** {d.metadata.get('source','')}")
                                st.write(d.page_content[:1200] + ("..." if len(d.page_content) > 1200 else ""))

        # ---------- Google Sheets logging ----------
        session_id = st.session_state.session_id
        models_selected = ", ".join(llm_choices)
        file_name = ""
        try:
            # If you’re holding uploaded PDFs in a variable like `uploaded_files`
            # adapt this line to your variable name:
            if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
                file_name = ", ".join([uf.name for uf in st.session_state.uploaded_files])
            elif 'last_indexed_files' in st.session_state:
                file_name = ", ".join(st.session_state.last_indexed_files)
        except Exception:
            pass

        # Build rows (one per model)
        rows_to_log = []
        for r in results:
            rows_to_log.append({
                "session_id": session_id,
                "file_name": file_name,
                "models_selected": models_selected,
                "prompt": q,
                "model_name": r["model"],
                "model_output": r.get("text", ""),
                "metrics": r.get("stats", {}),
                # date & device auto-filled in logger
            })

        # Append-only log
        log_rows(rows_to_log)


