from contextlib import asynccontextmanager, suppress
import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
import mlflow.pyfunc

from api.v1 import model_calc
from core.config import settings

from services.minio import MinioService
from models.config import AntifraudConfig





async def refresh_config_loop(app: FastAPI) -> None:
    while True:
        try:
            with MinioService() as minio:
                cfg = await minio.get_config(
                    object_name=settings.CONFIG_KEY,
                    model=AntifraudConfig,
                )
            app.state.antifraud_config = cfg
        except Exception as e: 
            print("ERROR refreshing config:", e)
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # try:
    #     mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    #     model = mlflow.pyfunc.load_model(settings.MODEL_URI)
    #     if model is None:
    #         raise RuntimeError(f"mlflow.pyfunc.load_model returned None for {settings.MODEL_URI}")
    #     app.state.model = model
    # except Exception as e:
    #     import traceback
    #     traceback.print_exc()
    #     app.state.model = None

    app.state.antifraud_config = None
    app.state.config_task = asyncio.create_task(refresh_config_loop(app))

    try:
        yield
    finally:
        task: asyncio.Task = app.state.config_task
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title=settings.PROJECT_NAME,
    docs_url='/api/openapi',
    openapi_url='/api/openapi.json',
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return ORJSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


app.include_router(model_calc.router, prefix='/api/v1', tags=['model_calc'])