import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://mlflow:5000")  # твой URI, как в сервисе

client = MlflowClient()

client.transition_model_version_stage(
    name="fraud_detector_s3",
    version="1",          # номер версии из UI
    stage="Production",   # нужный stage
)

print("OK")
