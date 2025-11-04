import time
import requests
from typing import Any, Dict, Tuple

DEFAULT_TIMEOUT = 120  # seconds

def generate_with_stats(
    base_url: str,
    model: str,
    prompt: str,
    options: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Use Ollama's /api/generate (non-streaming) to get text + detailed metrics.
    Returns (text, stats_dict).
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()

    # Text
    text = data.get("response", "")

    # Metrics from Ollama
    stats = {
        "latency_wall_s": dt,  # measured here (wall time)
        "prompt_eval_count": data.get("prompt_eval_count"),
        "prompt_eval_duration_s": (data.get("prompt_eval_duration") or 0) / 1e9,
        "eval_count": data.get("eval_count"),
        "eval_duration_s": (data.get("eval_duration") or 0) / 1e9,
        "total_duration_s": (data.get("total_duration") or 0) / 1e9,
        "load_duration_s": (data.get("load_duration") or 0) / 1e9,
    }
    # Derived
    if (stats["prompt_eval_count"] or 0) + (stats["eval_count"] or 0) > 0 and stats["total_duration_s"]:
        stats["tokens_per_s"] = (
            ((stats["prompt_eval_count"] or 0) + (stats["eval_count"] or 0)) / stats["total_duration_s"]
        )
    return text, stats


def get_model_info(base_url: str, model: str) -> Dict[str, Any]:
    """
    Returns model card-ish info from /api/show.
    Fields vary by model; we normalize a few.
    """
    url = f"{base_url.rstrip('/')}/api/show"
    r = requests.post(url, json={"name": model}, timeout=10)
    r.raise_for_status()
    data = r.json() or {}

    details = data.get("details", {}) or {}
    # Normalize some common fields if present
    return {
        "family": details.get("family"),
        "context_length": details.get("context_length") or details.get("context", None),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "format": details.get("format"),
        "raw": details,  # keep original in case you want to inspect later
    }
