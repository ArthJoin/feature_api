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

    MINIO_BUCKET_NAME: str = os.getenv("MINIO_BUCKET_NAME", "config")
    MINIO_ENDPOINT_URL: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9001")
    MINIO_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY_ID", "mlflow_access")
    MINIO_SECRET_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "mlflow_secret")
    CONFIG_KEY: str = os.getenv("CONFIG_KEY", "configs/antifraud_txn_v1.yaml")

    class Config:
        env_file = ".env"


settings = Settings()

logging_config.dictConfig(LOGGING)