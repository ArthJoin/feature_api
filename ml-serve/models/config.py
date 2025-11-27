from typing import Optional, Literal, List
from pydantic import BaseModel, field_validator, model_validator


class RetryConfig(BaseModel):
    max_retries: int
    backoff_ms: int


class TimeoutsConfig(BaseModel):
    scoring_ms: int
    total_request_ms: int
    retries: RetryConfig


class SLAConfig(BaseModel):
    max_error_rate: float
    max_p95_latency_ms: int


class LoggingConfig(BaseModel):
    logs_enabled: bool
    log_features: bool
    log_predictions: bool
    monitoring_enabled: bool
    sampling_rate: float


class TrafficConfig(BaseModel):
    stickiness: str  # "cst_txn_id" и т.п.


class ModelVariant(BaseModel):
    name: str
    role: Literal["decision", "shadow"]
    provider: Literal["mlflow"]
    registry_model_name: str
    registry_stage: Optional[str] = None
    version: int

    # новый конфиг
    feature_cols: List[str]
    cat_feature_names: List[str] = []

    weight: Optional[int] = None
    shadow_for: Optional[str] = None

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, value, info):
        role = info.data.get("role")
        if role == "decision" and value is None:
            raise ValueError("weight is required for decision models")
        if role == "shadow" and value is not None:
            raise ValueError("shadow model must not have weight")
        return value

    @field_validator("shadow_for")
    @classmethod
    def validate_shadow_for(cls, value, info):
        role = info.data.get("role")
        if role == "shadow" and value is None:
            raise ValueError("shadow model must define shadow_for")
        if role != "shadow" and value is not None:
            raise ValueError("shadow_for is allowed only for shadow models")
        return value

    @model_validator(mode="after")
    def validate_cat_features_subset(self):
        missing = set(self.cat_feature_names) - set(self.feature_cols)
        if missing:
            raise ValueError(
                f"missing in feature_cols: {sorted(missing)}"
            )
        return self


class AntifraudConfig(BaseModel):
    id: str
    enabled: bool

    traffic: TrafficConfig
    models: List[ModelVariant]

    timeouts: TimeoutsConfig
    sla: SLAConfig
    logging: LoggingConfig
