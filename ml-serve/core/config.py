import os
from typing import Optional
from pydantic_settings import BaseSettings
from logging import config as logging_config

from core.logger import LOGGING


class Settings(BaseSettings):
    # Название проекта
    PROJECT_NAME: str = "movies"

    # Корень проекта
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Annotated fields so Pydantic doesn't raise about non-annotated attributes
    MODEL_URI: Optional[str] = os.getenv("MODEL_URI")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns")

    class Config:
        env_file = ".env"


settings = Settings()

logging_config.dictConfig(LOGGING)