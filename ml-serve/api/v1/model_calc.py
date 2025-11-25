from fastapi import APIRouter, HTTPException, Body, Request
from typing import Any

from models.response import Response
from core.config import settings

import mlflow
import pandas as pd

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

router = APIRouter()

model = None


# @router.on_event("startup")
# def load_model() -> None:
#     global model
#     tracking_uri = settings.MLFLOW_TRACKING_URI
#     model_uri = settings.MODEL_URI

#     print(">>> Using MLFLOW_TRACKING_URI:", tracking_uri)
#     print(">>> Using MODEL_URI:", model_uri)

#     mlflow.pyfunc.mlflow.set_tracking_uri(tracking_uri)
#     model = mlflow.pyfunc.load_model(model_uri)

#     print("✅ Model loaded:", model)


@router.post("/score")
def score(request: Request, payload: Any = Body(...)):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        if not isinstance(payload, dict):
            return {"error": "Payload must be a JSON object"}

        features = payload.get("features")
        if features is None or not isinstance(features, dict):
            return {"error": "'features' field must be an object"}

        df = pd.DataFrame([features])

        pred = model.predict(df)

        if hasattr(pred, "tolist"):
            pred = pred.tolist()

        return {
            "prediction": pred
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))