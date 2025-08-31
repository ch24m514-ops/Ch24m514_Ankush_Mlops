import argparse
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Path to test parquet file")
    args = parser.parse_args()

    # Load best run info
    with open("artifacts/best_run.json", "r") as f:
        best_run = json.load(f)

    model_path = best_run["model_path"]
    print(f"📂 Loading model from {model_path}")
    model = joblib.load(model_path)

    test_df = pd.read_parquet(args.test)
    X_test = test_df.drop("Survived", axis=1)
    y_test = test_df["Survived"]

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"📊 Evaluation Results")
    print(f"✅ Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # ---- Save metrics for DVC ----
    metrics = {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist()
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("📑 Metrics saved to metrics.json")

if __name__ == "__main__":
    main()
