from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):

    APP_NAME: str = "Biddix AI Product Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    MAX_TURNS: int = 10
    GEMINI_API_KEY: str
    # GEMINI_MODEL: str = "this-model-does-not-exist"
    GEMINI_MODEL: str = "gemini-3.5-flash"


    HARD_TIMEOUT_SECONDS: int = 20

    API_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/models"

    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Maximum time allowed while establishing
    # a Redis connection.
    REDIS_CONNECT_TIMEOUT_SECONDS: float = 2.0

    # Maximum time allowed while waiting for
    # a Redis command response.
    REDIS_SOCKET_TIMEOUT_SECONDS: float = 2.0

    # first attempt + one retry
    # Total maximum attempts = 2
    REDIS_RETRY_ATTEMPTS: int = 1

    # Small pause before the retry.
    REDIS_RETRY_DELAY_SECONDS: float = 0.2

    # Session timeout of 24 hours
    SESSION_TTL_SECONDS: int = 86400
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60
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