import sys

from langchain_ollama import OllamaEmbeddings

from src.logger import logging
from src.exception import MyException


class OllamaEmbedder:
    """Embedding wrapper using a local Ollama model."""

    def __init__(self, model_name: str = "embeddinggemma"):
        self.model_name = model_name
        self._embedder = None

    def get_embedder(self) -> OllamaEmbeddings:
        """Create and return the Ollama embedding model."""
        if self._embedder is None:
            try:
                logging.info("Initializing the Ollama embedder.")
                self._embedder = OllamaEmbeddings(
                    model=self.model_name
                )
            except Exception as e:
                raise MyException(e, sys)

        return self._embedder
