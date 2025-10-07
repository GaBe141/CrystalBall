# Spark forecasting model wrapper
# Requires PySpark and Spark MLlib
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession


def train_spark_linear(df_path, target_col, feature_cols, forecast_horizon=12):
    spark = SparkSession.builder.appName("CrystalBallSpark").getOrCreate()
    df = spark.read.csv(df_path, header=True, inferSchema=True)
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(df)
    lr = LinearRegression(featuresCol="features", labelCol=target_col)
    model = lr.fit(data)
    # For demonstration, just predict on the last N rows
    predictions = model.transform(data).select(target_col, "prediction").tail(forecast_horizon)
    spark.stop()
    return predictions

# Example usage:
# preds = train_spark_linear('data/processed/mydata.csv', 'target', ['feat1', 'feat2'], forecast_horizon=12)
