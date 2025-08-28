import argparse, json, os, yaml, mlflow, mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

MODEL_NAME = "titanic_model"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    label_col = params["train"]["label_col"]

    spark = SparkSession.builder.appName("titanic-train").getOrCreate()

    # MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("titanic_spark")

    train = spark.read.parquet(args.train)
    test  = spark.read.parquet(args.test)

    lr = LogisticRegression(featuresCol="features", labelCol=label_col, maxIter=params["train"]["max_iter"])

    grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [float(x) for x in params["train"]["regParam"]])
        .addGrid(lr.elasticNetParam, [float(x) for x in params["train"]["elasticNetParam"]])
        .build()
    )

    evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")

    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=int(params["train"]["cv_folds"]),
        parallelism=2,
        seed=42
    )

    with mlflow.start_run() as run:
        cv_model = cv.fit(train)
        best = cv_model.bestModel
        auc_train = evaluator.evaluate(best.transform(train))
        auc_test  = evaluator.evaluate(best.transform(test))

        mlflow.log_param("algorithm", "Spark-LogisticRegression")
        mlflow.log_metric("auc_train", auc_train)
        mlflow.log_metric("auc_test", auc_test)

        # Log the Spark model
        mlflow.spark.log_model(best, artifact_path="model", registered_model_name=MODEL_NAME)

        # Save run info for downstream stages
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/best_run.json","w") as f:
            json.dump({"run_id": run.info.run_id, "experiment_id": run.info.experiment_id}, f)

        print(f"Logged to MLflow. AUC train={auc_train:.4f}, test={auc_test:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()
