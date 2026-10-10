from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Biddix AI Product Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3.5-flash"
    HARD_TIMEOUT_SECONDS: int = 20
    API_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/models"

    SHOPIFY_STORE_DOMAIN: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()