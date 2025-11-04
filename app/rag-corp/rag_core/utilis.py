from typing import Iterable
def ensure_models_note(model_name: str) -> str:
    return f"Make sure you have pulled the model: `ollama pull {model_name}`"
