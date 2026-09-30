"""Configuration settings for the Website-Grounded RAG Agent."""

import os
from pathlib import Path
from typing import Dict
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chromadb"
    CRAWLED_DATA_FILE: Path = DATA_DIR / "crawled_pages.json"

    # Crawler Settings
    SEED_URL: str = "https://fastapi.tiangolo.com/tutorial/"
    ALLOWED_DOMAIN: str = "fastapi.tiangolo.com"
    URL_PATH_PREFIX: str = "/tutorial/"
    MAX_PAGES: int = 50
    CRAWL_DELAY_SECONDS: float = 0.2
    CRAWL_TIMEOUT_SECONDS: int = 10
    USER_AGENT: str = (
        "WebsiteGroundedRAGBot/1.0 (+https://github.com/assessment/rag-agent)"
    )

    # Text Splitting & Chunking
    CHUNK_SIZE: int = 1200  # characters
    CHUNK_OVERLAP: int = 200  # characters

    # Vector Store & Embeddings
    # Options: "fastembed" (local ONNX, zero cost), "openai"
    EMBEDDING_PROVIDER: str = "fastembed"
    FASTEMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    COLLECTION_NAME: str = "website_knowledge_base"

    # LLM Settings
    # Options: "openai", "local_extractive" (offline fallback)
    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0

    # Retrieval Settings
    TOP_K: int = 4
    SIMILARITY_SCORE_THRESHOLD: float = 0.30

    # Pricing Models (per 1,000,000 tokens in USD)
    PRICING: Dict[str, Dict[str, float]] = {
        "gpt-4o-mini": {
            "input": 0.150,  # $0.15 per 1M input tokens
            "output": 0.600,  # $0.60 per 1M output tokens
        },
        "gpt-4o": {
            "input": 2.500,  # $2.50 per 1M input tokens
            "output": 10.000,  # $10.00 per 1M output tokens
        },
        "text-embedding-3-small": {
            "input": 0.020,  # $0.02 per 1M tokens
            "output": 0.0,
        },
        "BAAI/bge-small-en-v1.5": {
            "input": 0.0,  # Local, 0 cost
            "output": 0.0,
        },
    }


settings = Settings()
