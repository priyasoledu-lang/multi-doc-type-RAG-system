
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Generate embeddings locally without an API key."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self._embedder = None

    def get_embedder(self):
        """Load the embedding model once and reuse it."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(
                self.model_name
            )
        return self._embedder
