import tempfile, shutil, logging
from langchain_community.document_loaders import GitLoader

logger = logging.getLogger(__name__)

def load_repo(clone_url: str, branch: str, include_exts: list[str], exclude_dirs: list[str]):
    tmp_dir = tempfile.mkdtemp(prefix="repo_")
    try:
        loader = GitLoader(
            repo_path=tmp_dir,
            clone_url=clone_url,
            branch=branch,
            file_filter=lambda p: (any(p.endswith(ext) for ext in include_exts)
                                   and not any(f"/{d}/" in p for d in exclude_dirs))
        )
        docs = loader.load()
        logger.info("Cloned repo and loaded %d files.", len(docs))
        return docs
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
