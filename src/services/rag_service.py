"""RAG (Retrieval-Augmented Generation) service for SVY Agent."""
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from config import config
from src.utils.logging_config import logger


class RAGService:
    """RAG service for handling embeddings, vector store, and LLM queries."""

    def __init__(self):
        self.llm: Optional[ChatOpenAI] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self.chain: Optional[ConversationalRetrievalChain] = None
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize all RAG components."""
        logger.info("Initializing RAG service components...")
        
        self._load_embeddings()
        self._load_vector_store()
        self._initialize_llm()
        self._build_retrieval_chain()
        
        self._is_initialized = True
        logger.info("RAG service initialized successfully")

    def _load_embeddings(self) -> None:
        """Load the HuggingFace embeddings model."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                model_kwargs={"device": config.embedding_device}
            )
            logger.info(f"Embeddings model loaded: {config.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {e}")
            raise RuntimeError(f"Embeddings initialization failed: {e}") from e

    def _load_vector_store(self) -> None:
        """Load the FAISS vector store from disk."""
        if not os.path.exists(config.index_dir):
            raise RuntimeError(f"Vector store directory not found: {config.index_dir}")
        
        try:
            self.vector_store = FAISS.load_local(
                folder_path=config.index_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from: {config.index_dir}")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise RuntimeError(f"Vector store initialization failed: {e}") from e

    def _initialize_llm(self) -> None:
        """Initialize the OpenAI LLM."""
        try:
            self.llm = ChatOpenAI(
                model=config.llm_model,
                openai_api_key=config.openai_api_key,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
            logger.info(f"LLM initialized: {config.llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}") from e

    def _build_retrieval_chain(self) -> None:
        """Build the conversational retrieval chain."""
        system_prompt = (
            "You are SVY AGENT, an expert AI specializing in Geomatics-related topics. "
            "Answer user questions with clear, accurate, and concise explanations. "
            "Use a professional yet approachable tone."
        )
        
        prompt_template = PromptTemplate(
            input_variables=["system_prompt", "context", "question"],
            template=(
                "{system_prompt}\n\n"
                "Relevant Context:\n{context}\n\n"
                "Human: {question}\n\n"
                "Assistant:"
            )
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        try:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": config.retrieval_k}
                ),
                memory=memory,
                combine_docs_chain_kwargs={
                    "prompt": prompt_template.partial(system_prompt=system_prompt)
                },
                return_source_documents=True
            )
            logger.info("Retrieval chain created successfully")
        except Exception as e:
            logger.error(f"Failed to create retrieval chain: {e}")
            raise RuntimeError(f"Chain initialization failed: {e}") from e

    @property
    def is_ready(self) -> bool:
        """Check if the RAG service is ready to handle queries."""
        return self._is_initialized and self.chain is not None

    def query(self, message: str, debug: bool = False) -> dict:
        """Query the RAG system.
        
        Args:
            message: User message/query
            debug: Whether to include source documents in response
            
        Returns:
            Dictionary with answer and optionally source documents
        """
        if not self.is_ready:
            raise RuntimeError("RAG service is not initialized")
        
        # Log retrieved documents for debugging
        if debug:
            self._log_retrieved_documents(message)
        
        try:
            response = self.chain.invoke({"question": message})
            answer = response.get("answer", response)
            
            result = {"answer": answer}
            
            if debug:
                source_docs = response.get("source_documents", [])
                serialized_sources = self._serialize_sources(source_docs)
                result["sources"] = serialized_sources
            
            return result
            
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            raise

    def _log_retrieved_documents(self, query: str) -> None:
        """Log retrieved documents for debugging purposes."""
        try:
            top_docs = self.vector_store.similarity_search_with_score(query, k=config.retrieval_k)
            logger.debug(f"===== Retrieved Top-{config.retrieval_k} Documents =====")
            for rank, (doc, score) in enumerate(top_docs, start=1):
                meta = getattr(doc, "metadata", {}) if doc else {}
                pdf_name = meta.get("pdf_name", "Unknown PDF")
                chunk_id = meta.get("chunk_id", "unknown_chunk")
                snippet = (doc.page_content[:300] + "...") if getattr(doc, "page_content", None) else "<no content>"
                logger.debug(f"#{rank} score={score:.4f} source={pdf_name} chunk={chunk_id}")
                logger.debug(f"Content: {snippet}")
            logger.debug("===== End Retrieved Documents =====")
        except Exception as e:
            logger.warning(f"Similarity search logging failed: {e}")

    @staticmethod
    def _serialize_sources(source_docs: list) -> list:
        """Serialize source documents for JSON response."""
        serialized = []
        for doc in source_docs:
            meta = getattr(doc, "metadata", {})
            serialized.append({
                "pdf_name": meta.get("pdf_name", "Unknown PDF"),
                "chunk_id": meta.get("chunk_id", "unknown_chunk"),
                "content_preview": getattr(doc, "page_content", "")[:500]
            })
        return serialized
