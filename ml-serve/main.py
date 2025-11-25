from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
import mlflow.pyfunc

from api.v1 import model_calc
from core.config import settings

import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация при старте приложения
    print(">>> Tracking URI:", settings.MLFLOW_TRACKING_URI)
    print(">>> MODEL URI:", settings.MODEL_URI)

    
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        model = mlflow.pyfunc.load_model(settings.MODEL_URI)
        if model is None:
            raise RuntimeError(f"mlflow.pyfunc.load_model returned None for {settings.MODEL_URI}")
        app.state.model = model
        print(">>> Model loaded into app.state; type:", type(model))
    except Exception as e:
        print(">>> ERROR loading model:", e)
        import traceback
        traceback.print_exc()
        app.state.model = None

    yield  

app = FastAPI(
    title=settings.PROJECT_NAME,
    docs_url='/api/openapi',
    openapi_url='/api/openapi.json',
    default_response_class=ORJSONResponse,
    lifespan=lifespan,  # передаём lifespan здесь
)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return ORJSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


app.include_router(model_calc.router, prefix='/api/v1', tags=['model_calc'])