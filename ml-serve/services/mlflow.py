# services/mlflow_service.py
from __future__ import annotations

from typing import Dict, Any, List

import mlflow
import mlflow.pyfunc
import pandas as pd

from core.config import settings
from models.config import AntifraudConfig, ModelVariant


class MLflowService:
    def __init__(self, config: AntifraudConfig, tracking_uri: str | None = None) -> None:
        self._config = config
        self._tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(self._tracking_uri)

        self._models: Dict[str, mlflow.pyfunc.PyFuncModel] = {}
        self._load_all_models()

    @staticmethod
    def _build_model_uri(variant: ModelVariant) -> str:
        return f"models:/{variant.registry_model_name}/{variant.version}"

    def _load_all_models(self) -> None:
        for variant in self._config.models:
            uri = self._build_model_uri(variant)
            model = mlflow.pyfunc.load_model(uri)
            if model is None:
                raise RuntimeError(f"mlflow.pyfunc.load_model returned None for URI={uri}")
            self._models[variant.name] = model

    def predict(self, model_name: str, features: dict | List[dict]) -> Any:
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not loaded from config")

        model = self._models[model_name]

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = pd.DataFrame(features)

        return model.predict(df)
    
    def reorder(self, features_list, features_cols) -> List[dict]:
        reordered: List[dict] = []

        for idx, features in enumerate(features_list):

            new_row: Dict[str, Any] = {}
            missing_cols = []

            for col in features_cols:
                if col not in features:
                    missing_cols.append(col)
                else:
                    new_row[col] = features[col]

            reordered.append(new_row)

        return reordered
