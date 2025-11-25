FROM ghcr.io/mlflow/mlflow:latest

RUN pip install --no-cache-dir \
    mlflow[extras] \
    psycopg2-binary \
    boto3

CMD ["bash", "-c", "mlflow --help"]