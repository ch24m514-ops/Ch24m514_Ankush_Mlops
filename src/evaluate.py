import argparse, json, mlflow
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    args = ap.parse_args()

    spark = SparkSession.builder.appName("titanic-eval").getOrCreate()
    with open("artifacts/best_run.json") as f:
        info = json.load(f)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    run_id = info["run_id"]
    model_uri = f"runs:/{run_id}/model"

    # Load spark model via MLflow
    model = mlflow.spark.load_model(model_uri)
    test = spark.read.parquet(args.test)

    evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")
    auc = evaluator.evaluate(model.transform(test))

    with open("metrics.json","w") as f:
        json.dump({"auc_test": auc}, f, indent=2)

    spark.stop()
    print(f"Evaluation complete. AUC={auc:.4f} (written to metrics.json)")

if __name__ == "__main__":
    main()
