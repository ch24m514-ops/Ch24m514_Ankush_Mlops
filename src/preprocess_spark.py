import argparse, json, os, yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
import numpy as np

def compute_feature_stats(df, cols):
    stats = {}
    for c in cols:
        if dict(df.dtypes)[c] in ["int", "bigint", "double", "float", "long"]:
            s = df.select(F.mean(c).alias("mean"), F.stddev(c).alias("std"), F.count(c).alias("n")).collect()[0]
            stats[c] = {"mean": float(s["mean"]) if s["mean"] is not None else None,
                        "std": float(s["std"]) if s["std"] is not None else None,
                        "n": int(s["n"])}
        else:
            vc = df.groupBy(c).count().orderBy(F.col("count").desc())
            stats[c] = {"value_counts": {r[c]: int(r["count"]) for r in vc.collect()}}
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    label_col = params["train"]["label_col"]

    spark = SparkSession.builder.appName("titanic-preprocess").getOrCreate()

    df = spark.read.csv(args.input, header=True, inferSchema=True)

    # Basic cleaning
    df = df.drop("Name", "Ticket", "Cabin")  # drop high-cardinality/unnecessary cols
    df = df.withColumn("Age", F.when(F.col("Age").isNull(), F.avg("Age").over()).otherwise(F.col("Age")))
    df = df.na.fill({"Embarked": "S"})

    # Cast label to integer
    df = df.withColumn(label_col, F.col(label_col).cast("int"))

    # Categorical & numeric columns
    cats = ["Sex", "Embarked"]
    nums = ["Pclass","Age","SibSp","Parch","Fare"]

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cats]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_oh") for c in cats]

    assembler = VectorAssembler(
        inputCols=[*nums, *[f"{c}_oh" for c in cats]],
        outputCol="features"
    )

    pipeline = Pipeline(stages=[*indexers, *encoders, assembler])
    model = pipeline.fit(df)
    df_t = model.transform(df)

    # Train/test split
    train_df, test_df = df_t.randomSplit([0.8, 0.2], seed=42)

    os.makedirs(args.outdir, exist_ok=True)
    train_out = os.path.join(args.outdir, "train.parquet")
    test_out  = os.path.join(args.outdir, "test.parquet")
    train_df.select("features", label_col).write.mode("overwrite").parquet(train_out)
    test_df.select("features", label_col).write.mode("overwrite").parquet(test_out)

    # Save feature stats for drift monitoring
    os.makedirs("artifacts", exist_ok=True)
    stats = compute_feature_stats(df.select(*([label_col]+nums+cats)), nums+cats)
    with open("artifacts/train_feature_stats.json","w") as f:
        json.dump(stats, f, indent=2)

    # Persist the fitted preprocessing pipeline (to use in serving if needed)
    model.write().overwrite().save("artifacts/preprocess_pipeline")

    spark.stop()
    print(f"Wrote: {train_out}, {test_out}, artifacts/train_feature_stats.json, artifacts/preprocess_pipeline")

if __name__ == "__main__":
    main()
