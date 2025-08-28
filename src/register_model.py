import json, mlflow

MODEL_NAME = "titanic_model"

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    with open("artifacts/best_run.json") as f:
        info = json.load(f)

    run_id = info["run_id"]
    model_version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME).version
    print(f"Registered model version: {model_version}")

    # Transition to Production (archiving others)
    client = mlflow.MlflowClient()
    # Archive previous Production
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if mv.current_stage == "Production":
            client.transition_model_version_stage(name=MODEL_NAME, version=mv.version, stage="Archived")
    client.transition_model_version_stage(name=MODEL_NAME, version=model_version, stage="Production")
    print(f"Model {MODEL_NAME} v{model_version} transitioned to Production.")

if __name__ == "__main__":
    main()
