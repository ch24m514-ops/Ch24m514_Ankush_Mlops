import mlflow
import json
import os

MODEL_NAME = "titanic_model"

def main():
    # Load best run info
    with open("artifacts/best_run.json", "r") as f:
        best_run = json.load(f)

    model_path = best_run["model_path"]  # local path to joblib
    print(f"📂 Registering model from {model_path}")

    mlflow.set_tracking_uri("file:./mlruns")  # ensure consistent tracking dir
    mlflow.set_experiment("titanic-mlops")

    # Register the model directly from local path
    model_uri = f"file://{os.path.abspath(model_path)}"
    registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    print(f"✅ Successfully registered model '{MODEL_NAME}' (version {registered_model.version})")

if __name__ == "__main__":
    main()
