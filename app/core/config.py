from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application configuration settings."""
    MODEL_PATH: str = "models/model.pkl"
    DEBUG: bool = True

    class Config:
        env_file = "app/.env"

settings = Settings()
