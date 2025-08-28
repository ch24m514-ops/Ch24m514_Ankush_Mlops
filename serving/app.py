import os, mlflow, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Titanic Survival API", version="1.0.0")

MODEL_NAME = "titanic_model"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/Production"
    return mlflow.pyfunc.load_model(model_uri)

_model = None

@app.on_event("startup")
def _startup():
    global _model
    _model = load_model()

@app.post("/predict")
def predict(p: Passenger):
    global _model
    if _model is None:
        _model = load_model()

    df = pd.DataFrame([p.dict()])
    # Simple preprocessing to match training pipeline:
    # We'll call the Spark model via pyfunc which encapsulates the pipeline inside.
    # (The training step logged the Spark pipeline model.)

    pred = _model.predict(df)  # returns numpy array or series
    # For Spark-classifier pyfunc, prediction column is "prediction" or prob; use fallback
    try:
        y = float(pred[0])
    except Exception:
        # Fallback if dict-like
        y = float(pred[0].get("prediction", 0.0))
    return {"survived": int(round(y))}
