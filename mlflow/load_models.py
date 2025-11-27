import mlflow
from mlflow.models import infer_signature
from catboost import CatBoostClassifier
import pandas as pd
import json
import os

mlflow.set_tracking_uri("http://localhost:5000")

EXPERIMENT_NAME = "fraud_antifraud_experiment_s3"
MODEL_NAME = "fraud_detector_s3"

LOCAL_ARTIFACT_DIR = "/Users/arturnavruzov/Desktop/forte_fs_service/mlflow/m-a1c1873435df4f3986306afc75f6530f/artifacts"
MODEL_FILE = os.path.join(LOCAL_ARTIFACT_DIR, "model.cb")

exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    exp_id = mlflow.create_experiment(
        EXPERIMENT_NAME,
        artifact_location="s3://mlflow/fraud_antifraud_experiment_s3"
    )
else:
    exp_id = exp.experiment_id


with mlflow.start_run(experiment_id=exp_id) as run:

    model = CatBoostClassifier()
    model.load_model(MODEL_FILE)

    feature_order = model.feature_names_
    mlflow.set_tag("feature_order", json.dumps(feature_order))

    dummy_df = pd.DataFrame([ [0]*len(feature_order) ], columns=feature_order)
    preds = model.predict(dummy_df)

    signature = infer_signature(dummy_df, preds)

    mlflow.catboost.log_model(
        cb_model=model,
        artifact_path="model",
        # signature=signature,
        #input_example=dummy_df
    )

    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"Run ID: {run_id}")
    print(f"Model URI: {model_uri}")

mv = mlflow.register_model(model_uri, MODEL_NAME)

print(f"Registered model version: {mv.version}")
