import logging
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)

def get_llm(model: str, base_url: str):
    logger.info("Using LLM via Ollama: %s", model)
    return Ollama(model=model, base_url=base_url)
