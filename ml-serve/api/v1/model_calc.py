from fastapi import APIRouter, HTTPException, Body, Request
from typing import Any, Dict, List

from models.config import AntifraudConfig
from services.mlflow import MLflowService

router = APIRouter()


@router.post("/score")
async def score(request: Request, payload: Any = Body(...)) -> Dict[str, Any]:
    #getattr(request.app.state, "mlflow_service", None)
    config: AntifraudConfig = getattr(request.app.state, "antifraud_config", None)
    mlflow_service = MLflowService(config)

    if mlflow_service is None:
        raise HTTPException(status_code=503, detail="MLflow service not initialized")

    if config is None:
        raise HTTPException(status_code=503, detail="Config not loaded")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")

    features = payload.get("features")
    if features is None:
        raise HTTPException(status_code=400, detail="'features' field is required")

    if isinstance(features, dict):
        features_list: List[dict] = [features]
    elif isinstance(features, list) and all(isinstance(f, dict) for f in features):
        features_list = features
    else:
        raise HTTPException(
            status_code=400,
            detail="'features' must be an object or a list of objects",
        )
    
    reordered: List[dict] = mlflow_service.reorder(features_list, config.models[0].feature_cols)

    print("Reordered features:", reordered)
    print("list of feature cols:", features_list)

    results: Dict[str, Any] = {}

    for variant in config.models:
        model_name = variant.name
        try:
            pred = mlflow_service.predict(model_name=model_name, features=reordered)
            if hasattr(pred, "tolist"):
                pred = pred.tolist()
            results[model_name] = {
                "role": variant.role,
                "prediction": pred,
            }
        except Exception as e:
            results[model_name] = {
                "role": variant.role,
                "error": str(e),
            }

    return {
        "config_id": config.id,
        "models_count": len(config.models),
        "results": results,
    }
