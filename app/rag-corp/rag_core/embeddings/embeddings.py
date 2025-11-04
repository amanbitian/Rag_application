import logging
from typing import Optional

logger = logging.getLogger(__name__)

def _try_ollama(model: str, base_url: str):
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        emb = OllamaEmbeddings(model=model, base_url=base_url)
        probe = emb.embed_query("ok")
        if not probe:
            raise RuntimeError("Ollama embeddings returned empty vector.")
        logger.info("Using Ollama embeddings: %s", model)
        return emb
    except Exception as e:
        logger.warning("Ollama embeddings unavailable (%s): %s", model, e)
        return None

def _hf_fallback(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    logger.info("Using HF embeddings fallback: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True}
    )

def get_embedder(pref_model: str, base_url: str, hf_fallback_model: Optional[str] = None):
    """Return a working embeddings backend. Prefer Ollama; fallback to HF."""
    emb = _try_ollama(pref_model, base_url)
    if emb:
        return emb
    return _hf_fallback(hf_fallback_model or "sentence-transformers/all-MiniLM-L6-v2")
