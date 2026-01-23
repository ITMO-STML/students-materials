import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str

    # Cloud.ru OpenAI-compatible gateway
    base_url: str = "https://foundation-models.api.cloud.ru/v1"

    # Input materials
    resources_dir: str = "resources"

    # LLM models
    lector_model: str = "openai/gpt-oss-120b"
    enricher_model: str = "ai-sage/GigaChat3-10B-A1.8B"

    # Embeddings model
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # RAG params
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5


def load_settings() -> Settings:
    """Load .env and return typed settings.

    Required env vars:
      - API_KEY
    Optional:
      - BASE_URL (defaults to cloud.ru gateway)
      - RESOURCES_DIR
    """
    load_dotenv()

    api_key = os.environ["API_KEY"]
    base_url = os.environ.get("BASE_URL", Settings.base_url)
    resources_dir = os.environ.get("RESOURCES_DIR", Settings.resources_dir)

    return Settings(api_key=api_key, base_url=base_url, resources_dir=resources_dir)
