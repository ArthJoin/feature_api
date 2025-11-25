import mlflow
import os

# mlflow.set_tracking_uri("postgresql://mlflow:mlflow_pass@localhost:5434/mlflow")
# mlflow.set_experiment("fraud_antifraud_experiment")

mlflow.pyfunc.mlflow.set_tracking_uri("postgresql+psycopg2://mlflow:mlflow_pass@localhost:5434/mlflow")

with mlflow.start_run() as run:
    # Залогировать всю папку artifacts
    
   
    model = mlflow.pyfunc.load_model("models:/fraud_detector/2")

    print("Model loaded successfully:", model)