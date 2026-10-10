from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    APP_NAME: str = "Biddix AI Product Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3.5-flash"
    HARD_TIMEOUT_SECONDS: int = 20
    API_BASE_URL: str = (
        "https://generativelanguage.googleapis.com/v1beta/models"
    )

    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    SHOPIFY_STORE_DOMAIN: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


# Project paths

DATA_DIR = BASE_DIR / "data"

PRODUCT_FILE = DATA_DIR / "products.json"

VECTOR_DIR = DATA_DIR / "vector_store"

INDEX_FILE = VECTOR_DIR / "products.faiss"

METADATA_FILE = VECTOR_DIR / "products_metadata.json"