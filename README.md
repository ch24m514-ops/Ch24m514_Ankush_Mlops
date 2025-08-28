# Titanic MLOps Pipeline (Spark + DVC + MLflow + FastAPI)

End-to-end, production-style pipeline:
- **Data**: Titanic (downloaded from a public GitHub mirror)
- **Preprocessing**: Apache **Spark** (feature engineering + train/test split)
- **Versioning**: **DVC** (local remote preconfigured)
- **Experiment Tracking**: **MLflow** (SQLite backend, local artifact store)
- **Model**: Spark MLlib Logistic Regression with hyperparameter tuning
- **Registry & Deployment**: MLflow Model Registry → FastAPI inference service
- **Monitoring**: Simple drift detection (PSI) with auto-retrain trigger

## Quickstart

> Prereqs: Java 11+, Python 3.10+, PySpark, MLflow, DVC (already installed by your setup script).

```bash
# 0) Create and activate your venv if you haven't
source ~/mlops-env/bin/activate

# 1) Initialize DVC (first time only)
dvc init -f
dvc remote add -d localstore ./dvc_storage
git add . && git commit -m "init dvc"

# 2) Run the full pipeline
dvc repro

# 3) Start MLflow UI (optional)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# 4) Serve the model via FastAPI (reads latest Production model from registry)
uvicorn serving.app:app --host 0.0.0.0 --port 8000

# 5) Drift check + conditional retrain
python monitoring/drift.py --new_data data/new_batch.csv --threshold 0.2
```

## DVC Pipeline
```bash
dvc repro
```
Stages:
- **get_data** → downloads `data/raw/titanic.csv`
- **preprocess** → Spark job → `data/processed/train.parquet`, `data/processed/test.parquet`
- **train** → Spark MLlib + MLflow logging, registers model if best
- **evaluate** → logs metrics to MLflow and writes `metrics.json`
- **register** → promotes the best run to **Production** in MLflow registry

## API Usage
```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"Pclass":3,"Sex":"male","Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"}'
```

## Docker (optional for serving)
```bash
docker build -t titanic-api:latest .
docker run -p 8000:8000 --env MLFLOW_TRACKING_URI=sqlite:///mlflow.db -v $PWD:/app titanic-api:latest
```

> The container expects to see `mlflow.db` and the `mlruns/` directory mounted (we bind-mount the repo).

## Notes
- Tracking DB: `sqlite:///mlflow.db`
- Artifacts: local `mlruns/`
- Model name in registry: `titanic_model`
