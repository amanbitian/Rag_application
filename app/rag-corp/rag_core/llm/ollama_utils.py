import requests
from typing import List

def list_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Return available model tag names from Ollama (e.g., 'llama3.2:1b')."""
    url = f"{base_url}/api/tags"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
        data = r.json()
        names = []
        for m in data.get("models", []):
            name = m.get("name", "")
            if name:
                names.append(name)
        # de-duplicate while preserving order
        seen, out = set(), []
        for n in names:
            if n not in seen:
                seen.add(n); out.append(n)
        return out
    except Exception:
        # Safe fallback shortlist if the API is unavailable
        return ["llama3.2:1b", "tinyllama:1.1b", "deepseek-coder:6.7b-instruct"]
