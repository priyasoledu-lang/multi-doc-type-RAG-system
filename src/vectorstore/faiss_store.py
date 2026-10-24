
import sys

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from src.embedding.embedder import SentenceTransformerEmbedder
from src.exception import MyException


class SentenceTransformerLangChainEmbeddings(Embeddings):
    """Connect Sentence Transformers to LangChain and FAISS."""

    def __init__(self):
        self.model = SentenceTransformerEmbedder().get_embedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()


class FaissVectorStore:
    def __init__(self):
        self.embedder = SentenceTransformerLangChainEmbeddings()

    def create_vector_store(self, documents: list) -> FAISS:
        try:
            langchain_documents = []

            for i, doc in enumerate(documents):
                if not isinstance(doc, dict):
                    raise TypeError(
                        f"Document {i} must be a dictionary."
                    )

                if "text" not in doc or "metadata" not in doc:
                    raise ValueError(
                        f"Document {i} needs 'text' and 'metadata'."
                    )

                langchain_documents.append(
                    Document(
                        page_content=doc["text"],
                        metadata=doc["metadata"],
                    )
                )

            if not langchain_documents:
                raise ValueError("No documents to index.")

            return FAISS.from_documents(
                langchain_documents,
                self.embedder,
            )

        except Exception as e:
            raise MyException(e, sys) from e
