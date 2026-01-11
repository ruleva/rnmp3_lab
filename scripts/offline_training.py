from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import os

# Set HADOOP_HOME to avoid Windows error
os.environ['HADOOP_HOME'] = os.getcwd()

# Initialize Spark with more memory
spark = SparkSession.builder \
    .appName("DiabetesOfflineTraining") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

df = spark.read.csv('data/offline.csv', header=True, inferSchema=True)

print(f"Total records: {df.count()}")
df.printSchema()


def transform_data(dataframe):
    feature_cols = [col for col in dataframe.columns if col != 'Diabetes_binary']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    return Pipeline(stages=[assembler, scaler])


transform_pipeline = transform_data(df)
transform_model = transform_pipeline.fit(df)
df_transformed = transform_model.transform(df)
df_transformed = df_transformed.withColumnRenamed('Diabetes_binary', 'label')

train_df, test_df = df_transformed.randomSplit([0.8, 0.2], seed=42)

print(f"Training records: {train_df.count()}")
print(f"Test records: {test_df.count()}")

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

# Model 1: Logistic Regression
print("\n=== Training Logistic Regression ===")
lr = LogisticRegression(featuresCol="features", labelCol="label")
paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()

cv_lr = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid_lr, evaluator=evaluator, numFolds=3, seed=42)
cv_model_lr = cv_lr.fit(train_df)
predictions_lr = cv_model_lr.transform(test_df)
f1_lr = evaluator.evaluate(predictions_lr)
print(f"Logistic Regression F1 Score: {f1_lr}")

# Model 2: Random Forest
print("\n=== Training Random Forest ===")
rf = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

cv_rf = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid_rf, evaluator=evaluator, numFolds=3, seed=42)
cv_model_rf = cv_rf.fit(train_df)
predictions_rf = cv_model_rf.transform(test_df)
f1_rf = evaluator.evaluate(predictions_rf)
print(f"Random Forest F1 Score: {f1_rf}")

# Model 3: Gradient Boosted Trees
print("\n=== Training Gradient Boosted Trees ===")
gbt = GBTClassifier(featuresCol="features", labelCol="label", seed=42)
paramGrid_gbt = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [10, 20]) \
    .addGrid(gbt.maxDepth, [3, 5]) \
    .build()

cv_gbt = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid_gbt, evaluator=evaluator, numFolds=3, seed=42)
cv_model_gbt = cv_gbt.fit(train_df)
predictions_gbt = cv_model_gbt.transform(test_df)
f1_gbt = evaluator.evaluate(predictions_gbt)
print(f"Gradient Boosted Trees F1 Score: {f1_gbt}")

# Select best model
models = {
    'LogisticRegression': (cv_model_lr, f1_lr),
    'RandomForest': (cv_model_rf, f1_rf),
    'GradientBoostedTrees': (cv_model_gbt, f1_gbt)
}

best_model_name = max(models, key=lambda k: models[k][1])
best_model, best_f1 = models[best_model_name]

print(f"\n=== Best Model: {best_model_name} ===")
print(f"Best F1 Score: {best_f1}")

# Create models directory
os.makedirs('models', exist_ok=True)

# Save models
try:
    transform_model.write().overwrite().save('models/transform_pipeline')
    print("Transformation pipeline saved to models/transform_pipeline")

    best_model.bestModel.write().overwrite().save('models/best_model')
    print(f"Best model ({best_model_name}) saved to models/best_model")
except Exception as e:
    print(f"Warning: Could not save with Spark format: {e}")
    print("Models trained successfully but not persisted to disk")

spark.stop()