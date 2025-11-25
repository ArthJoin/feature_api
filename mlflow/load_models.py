import mlflow

# 1. Бьём в докерный MLflow
mlflow.set_tracking_uri("http://localhost:5000")

LOCAL_ARTIFACT_DIR = "/Users/arturnavruzov/Desktop/forte_fs_service/mlflow/mlruns/2/d4177c5e9b6f4a858184e54e4dfc059d/artifacts/model"

EXPERIMENT_NAME = "fraud_antifraud_experiment_s3"
MODEL_NAME = "fraud_detector_s3"

# 2. Создаём новый эксперимент с артефактами в S3
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    exp_id = mlflow.create_experiment(
        EXPERIMENT_NAME,
        artifact_location="s3://mlflow/fraud_antifraud_experiment_s3"
    )
else:
    exp_id = exp.experiment_id

print("Using experiment_id:", exp_id)

# 3. Логируем артефакты в НОВЫЙ эксперимент
with mlflow.start_run(experiment_id=exp_id) as run:
    mlflow.log_artifacts(LOCAL_ARTIFACT_DIR, artifact_path="model")

    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    print("New run:", run_id)
    print("Model URI:", model_uri)

    mv = mlflow.register_model(model_uri, MODEL_NAME)
    print("Registered model version:", mv.version)

"""
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9001   # порт UI/API MinIO на хосте

export AWS_ACCESS_KEY_ID=mlflow_access
export AWS_SECRET_ACCESS_KEY=mlflow_secret

"""