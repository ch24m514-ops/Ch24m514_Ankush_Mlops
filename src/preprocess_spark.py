import argparse
import os
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw Titanic CSV")
    parser.add_argument("--outdir", required=True, help="Output directory for processed data")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("TitanicPreprocess").getOrCreate()

    # Read raw dataset
    df = spark.read.csv(args.input, header=True, inferSchema=True)

    # ---- Handle missing values ----
    mean_age = df.select(F.mean(F.col("Age"))).collect()[0][0]
    df = df.withColumn("Age", F.when(F.col("Age").isNull(), mean_age).otherwise(F.col("Age")))

    mean_fare = df.select(F.mean(F.col("Fare"))).collect()[0][0]
    df = df.withColumn("Fare", F.when(F.col("Fare").isNull(), mean_fare).otherwise(F.col("Fare")))

    mode_embarked = df.groupBy("Embarked").count().orderBy(F.desc("count")).first()[0]
    df = df.fillna({"Embarked": mode_embarked})

    # ---- Encode categorical variables ----
    df = df.withColumn("Sex", F.when(F.col("Sex") == "male", 1).otherwise(0))
    df = df.withColumn(
        "Embarked",
        F.when(F.col("Embarked") == "S", 0)
         .when(F.col("Embarked") == "C", 1)
         .otherwise(2)
    )

    # ---- Select relevant features ----
    selected_cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    df = df.select(selected_cols)

    # ---- Train/Test Split ----
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    train_path = f"{args.outdir}/train.parquet"
    test_path = f"{args.outdir}/test.parquet"

    train_df.write.mode("overwrite").parquet(train_path)
    test_df.write.mode("overwrite").parquet(test_path)

    print(f"✅ Preprocessing complete. Saved train → {train_path}, test → {test_path}")

    # ---- Generate feature statistics on train set ----
    stats = {}
    for col in selected_cols:
        agg_exprs = [
            F.count(col).alias("count"),
            F.countDistinct(col).alias("unique"),
            F.min(col).alias("min"),
            F.max(col).alias("max"),
        ]
        if col != "Survived":  # don't calculate mean for target
            agg_exprs.append(F.mean(col).alias("mean"))

        summary = train_df.select(*agg_exprs).collect()[0].asDict()
        stats[col] = {k: str(v) for k, v in summary.items() if v is not None}

    os.makedirs("artifacts", exist_ok=True)
    stats_path = "artifacts/train_feature_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"📊 Feature stats saved to {stats_path}")

if __name__ == "__main__":
    main()
