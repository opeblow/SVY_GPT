"""Configuration management for SVY Agent."""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Application configuration."""

    # Telegram
    telegram_token: str

    # OpenAI
    openai_api_key: str

    # RAG Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    chunk_size: int = 1200
    chunk_overlap: int = 100
    retrieval_k: int = 5

    # LLM Configuration
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500

    # Paths
    index_dir: str = "faiss_index"
    pdf_directory: str = "ALL_PDF_FILES"

    # Debug
    debug: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not telegram_token:
            raise ValueError("TELEGRAM_BOT_TOKEN must be set in .env file")

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env file")

        debug = os.getenv("DEBUG", "false").lower() == "true"

        return cls(
            telegram_token=telegram_token,
            openai_api_key=openai_api_key,
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "5")),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
            index_dir=os.getenv("INDEX_DIR", "faiss_index"),
            pdf_directory=os.getenv("PDF_DIRECTORY", "ALL_PDF_FILES"),
            debug=debug,
        )


config = AppConfig.from_env()

