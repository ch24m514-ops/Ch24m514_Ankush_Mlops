import argparse
import os
import json
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pyspark.sql import SparkSession

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"📂 Loading train data from {args.train}")
    print(f"📂 Loading test data from {args.test}")

    spark = SparkSession.builder.appName("Train Titanic Model").getOrCreate()
    train_df = spark.read.parquet(args.train)
    test_df = spark.read.parquet(args.test)

    train_pd = train_df.toPandas()
    test_pd = test_df.toPandas()

    # Split features & labels
    X_train = train_pd.drop("Survived", axis=1)
    y_train = train_pd["Survived"]
    X_test = test_pd.drop("Survived", axis=1)
    y_test = test_pd["Survived"]

    print("🚀 Training RandomForest model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"✅ Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # --- MLflow Tracking ---
    mlflow.set_experiment("titanic-mlops")
    with mlflow.start_run() as run:
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(clf, "model")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

    # --- Save model locally ---
    model_path = os.path.join(args.outdir, "model.joblib")
    joblib.dump(clf, model_path)
    print(f"📦 Model saved to {model_path}")

    # --- Save metadata for evaluate.py ---
    best_run_info = {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "model_path": os.path.abspath(model_path),  # ✅ for joblib loading
        "model_uri": model_uri,                     # ✅ for MLflow loading
        "run_id": run_id
    }

    with open(os.path.join(args.outdir, "best_run.json"), "w") as f:
        json.dump(best_run_info, f, indent=4)

    print("📝 Best run info saved to artifacts/best_run.json")


if __name__ == "__main__":
    main()
