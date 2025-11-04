import logging
import sys

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once."""
    if getattr(setup_logging, "_configured", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    fmt = "[%(asctime)s] %(levelname)s %(name)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    setup_logging._configured = True