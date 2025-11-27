# Feature API — AntiFraud ML Scoring Platform  

Готовая инфраструктура и бизнес-процесс для процессинга банковских транзакций. Проект реализован в рамках AI-хакатон от Forte 2025. Основная проблема, которую решает проект не просто "найти мошенничество". Банку требуется система, которая в реальном времени объединит / добавит / заменит данные из разных источников, обогатит данные, прогонит модели и примет решение с минимальным latency. Все это с минимальным cost of maintenance.

**Camunda BPM → Go Service → Feature List → FastAPI → MLFlow Registry -> Prediction**.

Проект реализован в формате production-архитектуры, заточен под масштабирование и замену ML-моделей без даунтайма.

---

## Key Features

- **Feature Engineering**: вычисление поведенческих, временных и агрегированных признаков
- **Feature API Service (Go)**: быстрый low-latency сервис для отдачи фич из PostgreSQL. (REDIS later)
- **ML Service (FastAPI)**: скоринг CatBoost модели, прокси через MLFlow Registry.
- **ML Model Hot-Swap**: автоматическая смена модели без перезапуска сервиса.
- **CatBoost ML Model**: модель обученая на базовых и посчитанных фичах.
- **MinIO Object Store**: хранение версионированных конфигураций и fallback-конфигов.
- **Camunda BPM**: оркестрация бизнес-процесса.

---

## Архитектура

```mermaid
graph LR
    U["Client App"] --> GW["API Gateway"]

    subgraph "Feature Layer"
        FS["Feature Service (Go)"] --> PG[("PostgreSQL")]
    end

    subgraph "ML Layer"
        INF["ML Inference (FastAPI)"]
        MLFlow["MLFlow Registry"]
        MinIO[("MinIO Object Store")]
    end

    subgraph "Orchestration"
        Camunda["Camunda BPM"]
    end

    GW --> Camunda
    Camunda --> FS
    Camunda --> INF

    INF --> MLFlow
    INF --> MinIO

    MLFlow -->|Model Artifacts| INF
    PG --> FS
```

---

## Prject Structure

```
feature_api/
│
├── camunda/                    # BPMN diagrams + configs
│
├── docker/                     # Dockerfiles
│
├── etl/                        # Feature engineering pipelines
│   ├── convert_raw_to_csv.py
│   └── samples/                # Ignored in git
│
├── feature-service/            # Go service: feature fetching
│   ├── main.go
│   ├── internal/http/
│   └── internal/repository/
│
├── ml/                         # Ml training
│   ├── data/
│   ├── processed/
│   └── src/
│
├── ml-serve/                   # FastAPI ML Service
│   ├── api/v1/
│   ├── configs/                # File loaded to MINIO via load_config_minio.py (Временно здесь)
│   ├── core/
│   ├── models/
│   ├── services/
│   ├── tests/
│   ├── main.py
│   └── requirements.txt
│
├── mlflow/                     # Local MLFlow setup
│   ├── pg_data/
│   └── python_etl_files
│
├── postgres/                   # PG as mini DWH layer
│   ├── data/
│   ├── dump/
│   └── init/
│
├── docker-compose.yaml
└── README.md
```

---

## Tech Stack

### ML / Data

* CatBoost
* XGBoost
* SHAP
* MLFlow Registry
* SHAP / Eval metrics
* Pandas / NumPy
* PostgreSQL
* MinIO (S3 storage)

### Backend

* Go (feature service)
* GIN
* FastAPI (ML inference)
* Uvicorn
* Pydantic models

### DevOps

* Docker / docker-compose

### Orchestration

* Camunda BPMN workflows

---

## How Model Scoring Works

1. Camunda вызывает Feature API → получает агрегированные признаки.
2. Camunda передаёт признаки в ML-Serve.
3. FastAPI → запрашивает модель (на основе конфига в minio refresh 60s) из MLFlow по `model_name/version`.
4. Модель выполняет CatBoost-скоринг.
5. Результат возвращается в Camunda (>0.000 <1.000).
---

## ML Model Hot-Swap (Zero Downtime)

ML Serve сервис не хранит модель локально → он перераспрашивает MLFlow по:

```
models:/{.registry_model_name}/{.version}
```

Чтобы заменить модель:

```
mlflow.register_model("runs:/<new_run_id>/model", "fraud_detector_s3")
mlflow.transition_model_version_stage -m "fraud_detector_s3" --version <num> --stage Production
```

или обновление конфигурации в minio через etl-скрипт проверяющий корректность по pydantic моделям (future: GitOps подход с CI/CD)

Доступно: 
1. Запуск нескольких моделей 
2. Запуск моделей с ролями Shadow / Desision
3. Отключение моделей 
4. Переключение версий 
5. timeout / retry / backoff / sla / logging options

Сервис автоматически подхватит новую модель при следующем запросе.

---

## Механизм конфигураций в MinIO

В MinIO хранятся:

* конфиги инференса
* fallback версии модели
* параметры threshold
* маппинги фичей
* список моделей для расчета

Пример структуры:

```
configs/
  antifraud_txn_v1.json
```

Конфиги версионируются через включённое MinIO versioning.

---

## Running Locally

```bash
docker-compose up -d

# MacOS
brew install minio/stable/mc

# Windows
Invoke-WebRequest https://dl.min.io/client/mc/release/windows-amd64/mc.exe -OutFile mc.exe

# Вставить реальные креды, указанные в .env
mc alias set minio http://localhost:9000 minio_access_key minio_secret_key

# Создадим бакеты
mc mb minio/config
mc mb minio/mlflow

# Загрузим модель в MLFlow
cd mlflow 
python3 load_models.py 

# Загрузим конфигурацию
cd ..
cd ml-serve
python3 load_config_minio.py --file configs/antifraud_txn_v1.yaml
```

Поднимутся:

* PostgreSQL
* MLFlow + Postgres backend
* MinIO
* Feature Service (Go)
* ML-Serve (FastAPI)
* Camunda


## API Examples

### 1. Camunda

`POST http://localhost:8080/engine-rest/process-definition/key/Process_16ugzyw/start`

Body:

```json
{
  "variables": {
    "cst_dim_id": { "value": 2096229005, "type": "Long" },
    "transdate": { "value": "2025-03-04 00:00:00.000", "type": "String" },
    "transdatetime": { "value": "2025-03-04 17:41:57.000", "type": "String" },
    "amount": { "value": 4000.0, "type": "Double" },
    "docno": { "value": "8442", "type": "String" },
    "direction": { "value": 1, "type": "Integer" },
    "target": { "value": "b3a3d4a6006293195d998957d4f01e42", "type": "String" }
  }
}
```

Response: 

Результат отработки проверить в `http://127.0.0.1:8080/camunda`

### 2. Feature API

`POST http://127.0.0.1:9000/features`

Body:

```json
{
  "cst_dim_id": 2096229005,
  "transdate": "2025-03-04",
  "transdatetime": "2025-03-04 17:41:57.000",
  "amount": 4000.0,
  "docno": "8442",
  "direction": 1,
  "target": "b3a3d4a6006293195d998957d4f01e42"
}

```

Response:

```json
{
    "features": {
        "amount": 4000,
        "amount_bin": "ALL",
        "amount_clipped": 4000,
        //...
        "zscore_login_abs": 0.0267908985024482
    }
}
```

### 3. ML-Serve

`POST http://127.0.0.1:81/api/v1/score`

```json
{
    "features": {
        "amount": 4000,
        "amount_bin": "ALL",
        "amount_clipped": 4000,
        //...
        "zscore_login_abs": 0.0267908985024482
    }
}
```

Response:

```json
{
    "config_id": "antifraud_txn_v1",
    "models_count": 1,
    "results": {
        "fraud_detector_s3": {
            "role": "decision",
            "prediction": [
                0,32
            ]
        }
    }
}
```

## Future Improvements

* Streaming-фичи (Kafka + Flink/Spark Structured Streaming)
* Real-time feature store
* Ensemble моделей (CatBoost + LightGBM + anomaly detectors)
* Automated retraining pipeline
* Drift detection (data & concept drift)
* Full audit pipeline (граф связей, lineage, аудит)