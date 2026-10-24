from dotenv import load_dotenv

load_dotenv()

import os
import sys
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from src.exception import MyException
from src.ingestion.extractor import DocumentExtractor
from src.ingestion.loaders import DocumentLoader
from src.logger import logging
from src.preprocessing.clean_normalize import (
    DocumentNormalizationAndCleaning,
)
from src.preprocessing.chunking import DocumentChunker
from src.rag import prompts
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RerankMMRRetriever
from src.utils.main_utils import (
    num_tokens_from_string,
    load_configs,
    build_context,
    extract_sources,
)
from src.vectorstore.faiss_store import FaissVectorStore


class RAGPipeline:
    """Multi-document RAG pipeline with local embeddings and cloud LLM."""

    def __init__(self, config_dir: str = "configs"):
        try:
            self.config = load_configs(config_dir)

            # -------------------------------------------------
            # Ollama Cloud configuration
            # -------------------------------------------------
            gen_cfg = self.config.get("generation", {})

            api_key = os.getenv("OLLAMA_API_KEY")
            base_url = os.getenv(
                "OLLAMA_BASE_URL",
                "https://ollama.com",
            )
            model_name = gen_cfg.get(
                "llm_model",
                "gemma4:31b-cloud",
            )

            if not api_key:
                raise ValueError(
                    "OLLAMA_API_KEY is missing. "
                    "Configure it in Streamlit Cloud Secrets."
                )

            self.llm = ChatOllama(
                model=model_name,
                temperature=float(gen_cfg.get("temperature", 0.2)),
                base_url=base_url,
                client_kwargs={
                    "headers": {
                        "Authorization": f"Bearer {api_key}",
                    }
                },
                num_ctx=int(gen_cfg.get("num_ctx", 2048)),
                num_predict=int(
                    gen_cfg.get("max_output_tokens", 512)
                ),
            )

            # -------------------------------------------------
            # Retrieval configuration
            # -------------------------------------------------
            retr_cfg = self.config.get("retrieval", {})

            self.reranker = CrossEncoderReranker(
                model_name=retr_cfg.get("reranker_model")
            )

            self.vector_store_builder = FaissVectorStore()

            # Pipeline components
            self.extractor = DocumentExtractor()
            self.loader = DocumentLoader()
            self.normalizer = DocumentNormalizationAndCleaning()
            self.chunker = DocumentChunker()

            self.context_token_limit = int(
                self.config.get(
                    "stuff_context_token_limit",
                    1000,
                )
            )

            self.vector_store = None
            self.retriever = None
            self.indexed_documents = []

            logging.info("RAGPipeline initialized successfully.")

        except Exception as exc:
            logging.exception("Failed to initialize RAGPipeline.")
            raise MyException(exc, sys) from exc

    
  
    def prepare_vector_store(self):
        """Load, clean, chunk, and index configured documents."""
        try:
            documents = self.config.get("documents", [])

            if not documents:
                raise ValueError(
                    "No documents found in pipeline.config['documents']."
                )

            all_chunks = []

            for item in documents:
                # Streamlit supplies dictionaries containing name and path.
                if isinstance(item, dict):
                    document_path = item.get("path")
                    display_name = item.get("name", document_path)
                else:
                    document_path = str(item)
                    display_name = document_path

                if not document_path:
                    logging.warning(
                        "Skipping a document with no path: %s", item
                    )
                    continue

                logging.info("Processing document: %s", display_name)

                # 1. Load the file or URL.
                loaded_documents = self.loader.load_document(
                    document_path
                )

                if not loaded_documents:
                    logging.warning(
                        "No content loaded from %s", document_path
                    )
                    continue

                # 2. Extract text and metadata.
                extracted_documents = self.extractor.extract_document_info(
                    loaded_documents,
                    document_path,
                )

                if not extracted_documents:
                    logging.warning(
                        "No text extracted from %s", document_path
                    )
                    continue

                # 3. Clean and normalize text.
                cleaned_documents = (
                    self.normalizer.initialize_document_normalizer(
                        extracted_documents
                    )
                )

                # Ensure source metadata identifies the uploaded document.
                for doc in cleaned_documents:
                    metadata = doc.setdefault("metadata", {})
                    metadata["source"] = display_name
                    metadata.setdefault(
                        "source_file", display_name
                    )

                # 4. Split the cleaned text into chunks.
                chunks = self.chunker.chunk_document(
                    cleaned_documents,
                    target_chunk_size=500,
                    chunk_overlap=100,
                )

                all_chunks.extend(chunks)

                logging.info(
                    "Generated %s chunks from %s",
                    len(chunks),
                    display_name,
                )

            if not all_chunks:
                raise ValueError(
                    "No text chunks were generated. Check that the "
                    "uploaded files contain readable text and use "
                    "a supported file type."
                )

            # 5. Build the FAISS index.
            self.vector_store = (
                self.vector_store_builder.create_vector_store(
                    all_chunks
                )
            )

            self.indexed_documents = all_chunks

            logging.info(
                "FAISS index created successfully with %s chunks.",
                len(all_chunks),
            )

            return self.vector_store

        except Exception as exc:
            logging.exception("Failed to prepare FAISS vector store.")
            raise MyException(exc, sys) from exc


    # ---------------------------------------------------------
    # LLM generation
    # ---------------------------------------------------------
    def generate_response(
        self,
        question: str,
        context: str,
    ) -> str:
        """Generate an answer using retrieved document context."""

        try:
            system_prompt = (
                "You are a careful research assistant. "
                "Answer using the supplied document context. "
                "If the context does not contain the answer, "
                "say that the documents do not provide enough "
                "information. Do not invent citations."
            )

            user_prompt = (
                f"Document context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "Provide a clear, concise answer grounded in "
                "the document context."
            )

            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )

            return str(response.content)

        except Exception as exc:
            logging.exception("LLM response generation failed.")
            raise MyException(exc, sys) from exc

    # ---------------------------------------------------------
    # Document indexing
    # ---------------------------------------------------------
    def index_documents(
        self,
        documents: list[dict[str, Any]],
    ):
        """Create a FAISS index from cleaned document chunks.

        Each item must contain:
            {
                "text": "...",
                "metadata": {"source": "filename.pdf"}
            }
        """

        try:
            if not documents:
                raise ValueError(
                    "No document chunks were provided for indexing."
                )

            self.vector_store = (
                self.vector_store_builder.create_vector_store(
                    documents
                )
            )

            self.indexed_documents = documents

            logging.info(
                "Successfully indexed %s document chunks.",
                len(documents),
            )

            return self.vector_store

        except Exception as exc:
            logging.exception("Document indexing failed.")
            raise MyException(exc, sys) from exc

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------
    def retrieve_documents(
        self,
        question: str,
        k: int = 5,
    ) -> list[Document]:
        """Retrieve relevant document chunks from FAISS."""

        try:
            if self.vector_store is None:
                raise ValueError(
                    "No FAISS index exists. Index documents first."
                )

            documents = self.vector_store.similarity_search(
                question,
                k=k,
            )

            logging.info(
                "Retrieved %s chunks for the question.",
                len(documents),
            )

            return documents

        except Exception as exc:
            logging.exception("Document retrieval failed.")
            raise MyException(exc, sys) from exc

    # ---------------------------------------------------------
    # Context construction
    # ---------------------------------------------------------
    def prepare_context(
        self,
        documents: list[Document],
    ) -> str:
        """Combine retrieved chunks into a context string."""

        try:
            context_parts = []

            for index, document in enumerate(documents, start=1):
                source = document.metadata.get(
                    "source",
                    document.metadata.get(
                        "source_file",
                        "Unknown source",
                    ),
                )

                context_parts.append(
                    f"[Source {index}: {source}]\n"
                    f"{document.page_content}"
                )

            context = "\n\n".join(context_parts)

            # Keep the context bounded to reduce generation cost.
            if (
                num_tokens_from_string(context)
                > self.context_token_limit
            ):
                words = context.split()
                context = " ".join(
                    words[: self.context_token_limit]
                )

            return context

        except Exception as exc:
            logging.exception("Context preparation failed.")
            raise MyException(exc, sys) from exc

    # ---------------------------------------------------------
    # Question answering
    # ---------------------------------------------------------
    def ask(
        self,
        question: str,
        k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve document evidence and answer a question."""

        try:
            if not question or not question.strip():
                raise ValueError(
                    "Please enter a question."
                )

            retrieved = self.retrieve_documents(
                question.strip(),
                k=k,
            )

            if not retrieved:
                return {
                    "question": question,
                    "answer": (
                        "I could not find relevant information "
                        "in the indexed documents."
                    ),
                    "sources": [],
                }

            context = self.prepare_context(retrieved)
            answer = self.generate_response(
                question=question.strip(),
                context=context,
            )

            sources = []

            for document in retrieved:
                source = document.metadata.get(
                    "source",
                    document.metadata.get(
                        "source_file",
                        "Unknown source",
                    ),
                )

                if source not in sources:
                    sources.append(source)

            return {
                "question": question,
                "answer": answer,
                "sources": sources,
            }

        except Exception as exc:
            logging.exception("Question answering failed.")
            raise MyException(exc, sys) from exc

    # ---------------------------------------------------------
    # Indexed-document summary
    # ---------------------------------------------------------
    def get_indexed_documents_summary(
        self,
    ) -> dict[str, Any]:
        """Return basic information about indexed chunks."""
    
    def answer_with_sources(
        self,
        question: str,
        k: int = 5,
    ) -> dict[str, Any]:
        """Answer a question and return its supporting sources."""
        result = self.ask(question=question, k=k)
        source_details = []
        for source in result.get("sources", []):
            if isinstance(source, dict):
                source_details.append({
                    "path": source.get(
                        "path",
                        source.get(
                            "source",
                            source.get("source_file", "Unknown source"),
                        ),
                    ),
                    "name": source.get(
                        "name",
                        source.get("source", "Unknown source"),
                    ),
                })
            else:
                source_details.append({
                    "path": str(source),
                    "name": str(source),
                })

        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": source_details,
        }

        try:
            sources = set()

            for document in self.indexed_documents:
                metadata = document.get("metadata", {})
                source = metadata.get(
                    "source",
                    metadata.get(
                        "source_file",
                        "Unknown source",
                    ),
                )
                sources.add(source)

            return {
                "total_chunks": len(self.indexed_documents),
                "total_sources": len(sources),
                "sources": sorted(sources),
            }

        except Exception as exc:
            logging.exception(
                "Unable to summarize indexed documents."
            )
            raise MyException(exc, sys) from exc
