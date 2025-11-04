from dataclasses import dataclass
from pathlib import Path
import os, yaml
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    env: str
    data_dir: str
    index_dir: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    embed_model: str
    llm_model: str
    git_branch: str
    git_include_exts: list[str]
    git_exclude_dirs: list[str]
    ollama_base_url: str

def _candidate_paths() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # .../rag_core -> .../Rag_application/
    return [
        Path(os.getenv("CONFIG_PATH", "")).expanduser(),
        repo_root / "configs" / "settings.yaml",
        repo_root / "app" / "configs" / "settings.yaml",
    ]

def load_settings(path: str | None = None) -> Settings:
    #  Locate the config file
    candidates = _candidate_paths()
    cfg_path = Path(path).expanduser() if path else next((p for p in candidates if p and p.is_file()), None)
    if not cfg_path:
        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"settings.yaml not found. Tried:\n{tried}")

    # Load YAML safely
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)

    #⃣ Return dataclass instance
    return Settings(
        env=data.get("env", "dev"),
        data_dir=data.get("data_dir", "data"),
        index_dir=data.get("index_dir", "data/index/faiss"),
        chunk_size=data.get("chunk_size", 1000),
        chunk_overlap=data.get("chunk_overlap", 200),
        retriever_k=data.get("retriever_k", 4),
        embed_model=data.get("embed_model", "bge-m3"),
        llm_model=data.get("llm_model", "llama3.2:1b"),
        git_branch=data["git"]["branch"],
        git_include_exts=data["git"]["include_exts"],
        git_exclude_dirs=data["git"]["exclude_dirs"],
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
