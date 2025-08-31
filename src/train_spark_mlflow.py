import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train parquet file")
    parser.add_argument("--test", required=True, help="Path to test parquet file")
    parser.add_argument("--outdir", required=True, help="Directory to save artifacts")
    args = parser.parse_args()

    # ---- Load processed data ----
    print(f"📂 Loading train data from {args.train}")
    train_df = pd.read_parquet(args.train)

    print(f"📂 Loading test data from {args.test}")
    test_df = pd.read_parquet(args.test)

    X_train = train_df.drop("Survived", axis=1)
    y_train = train_df["Survived"]
    X_test = test_df.drop("Survived", axis=1)
    y_test = test_df["Survived"]

    # ---- Train model ----
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"✅ Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # ---- Log with MLflow ----
    mlflow.set_experiment("titanic-mlops")
    with mlflow.start_run() as run:
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # ---- Save model locally ----
        os.makedirs(args.outdir, exist_ok=True)
        model_path = os.path.join(args.outdir, "model.joblib")
        joblib.dump(model, model_path)
        print(f"📦 Model saved to {model_path}")

        # ---- Save best run metadata for evaluate stage ----
        best_run_info = {
            "run_id": run.info.run_id,
            "experiment_id": mlflow.get_experiment_by_name("titanic-mlops").experiment_id,
            "metrics": {"accuracy": acc, "f1_score": f1},
            "model_path": model_path
        }
        best_run_path = os.path.join(args.outdir, "best_run.json")
        with open(best_run_path, "w") as f:
            json.dump(best_run_info, f, indent=4)

        print(f"📝 Best run info saved to {best_run_path}")

if __name__ == "__main__":
    main()
