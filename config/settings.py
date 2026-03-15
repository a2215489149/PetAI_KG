from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "0301_petAI"
    VERSION: str = "2.1.0"

    # API Keys & URLs
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_API_VERSION: str = "2025-04-01-preview"
    AZURE_GPT4O_DEPLOYMENT: str = "gpt-4o"
    AZURE_EMBED_DEPLOYMENT: str = "text-embedding-3-small"
    
    # Gemini
    GEMINI_API_KEY: str = ""

    # LINE Bot Settings
    LINE_CHANNEL_ACCESS_TOKEN: str = ""
    LINE_CHANNEL_SECRET: str = ""
    EXPERT_REVIEW_GROUP_ID: Optional[str] = ""
    EXPERT_DIRECT_NOTIFY_USER_ID: Optional[str] = ""
    DEFAULT_TENANT: str = "line"

    # Database Settings
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    
    NEO4J_URI: str = ""
    NEO4J_USER: str = ""
    NEO4J_PASSWORD: str = ""
    
    REDIS_URL: str = ""
    
    POSTGRES_DSN: str = ""

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "pet_ai_system.log"

    class Config:
        env_file = ".env"

settings = Settings()
