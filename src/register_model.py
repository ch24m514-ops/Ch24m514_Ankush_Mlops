import mlflow
import json
import os

MODEL_NAME = "titanic_model"
MODEL_PATH = "artifacts/model.joblib"
URI_FILE = "artifacts/latest_model_uri.txt"

def main():
    print(f"📂 Registering model from {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}. Please run `dvc repro` first.")

    # Start MLflow run (ensures run_id is always valid)
    with mlflow.start_run(run_name="register_model") as run:
        run_id = run.info.run_id

        # Log model artifact to MLflow under this run
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")

        # Register model from this run
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

        print(f"Successfully registered model '{MODEL_NAME}'.")
        print(f"Created version '{result.version}' of model '{MODEL_NAME}'.")

        # Save URI to a file for later serving
        with open(URI_FILE, "w") as f:
            f.write(f"models:/{MODEL_NAME}/{result.version}")

        print(f"✅ Saved latest model URI to {URI_FILE}")
        print(f"💡 Serve with: mlflow models serve -m $(cat {URI_FILE}) -p 5000")


if __name__ == "__main__":
    main()
