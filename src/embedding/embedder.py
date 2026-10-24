
import sys
from langchain_openai import OpenAIEmbeddings
from src.logger import logging
from src.exception import MyException


class OllamaEmbedder:
    """
    Embedding wrapper using OpenAI for cloud deployment.
    The class name is retained to avoid breaking existing imports.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self._embedder = None

    def get_embedder(self) -> OpenAIEmbeddings:
        """Create and return the OpenAI embedding model."""
        if self._embedder is None:
            try:
                logging.info("Initializing the OpenAI embedder.")
                self._embedder = OpenAIEmbeddings(
                    model=self.model_name
                )
            except Exception as e:
                raise MyException(e, sys)

        return self._embedder
