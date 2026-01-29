# Databricks notebook source
# notebooks/03_train_model.py

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

ENV = dbutils.widgets.get("env")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
feature_pipeline = dbutils.widgets.get("feature_pipeline")

CATALOG = f"mlops_{ENV}"
SCHEMA = "feature_engineering"

# ----------------------------
# Experiment setup
# ----------------------------
# EXPERIMENT_NAME = "/Shared/mlops_dev"
mlflow.set_experiment(EXPERIMENT_NAME)

# ----------------------------
# Load feature pipeline
# ----------------------------
feature_pipeline = joblib.load(feature_pipeline)

# ----------------------------
# Load train/val data
# ----------------------------
train_sdf = spark.table(f"{CATALOG}.{SCHEMA}.train_split")
val_sdf = spark.table(f"{CATALOG}.{SCHEMA}.val_split")

train_df = train_sdf.toPandas()
val_df = val_sdf.toPandas()

TARGET_COL = "Potability"

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

X_val = val_df.drop(columns=[TARGET_COL])
y_val = val_df[TARGET_COL]

# ----------------------------
# Transform features
# ----------------------------
X_train_transformed = feature_pipeline.transform(X_train)
X_val_transformed = feature_pipeline.transform(X_val)

# ----------------------------
# Train model
# ----------------------------
model = LogisticRegression(max_iter=1000, class_weight="balanced")

with mlflow.start_run(run_name="logreg_pipeline") as run:
    run_id = run.info.run_id

    # Fit the model
    model.fit(X_train_transformed, y_train)

    # ----------------------------
    # Predictions & Metrics
    # ----------------------------
    y_pred = model.predict(X_val_transformed)
    y_prob = model.predict_proba(X_val_transformed)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_prob)

    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("val_roc_auc", roc)

    # ----------------------------
    # Log original datasets
    # ----------------------------
    train_dataset = mlflow.data.from_spark(train_sdf, table_name=f"{CATALOG}.{SCHEMA}.train_split")
    mlflow.log_input(train_dataset, context="training")

    val_dataset = mlflow.data.from_spark(val_sdf, table_name=f"{CATALOG}.{SCHEMA}.val_split")
    mlflow.log_input(val_dataset, context="validation")

    # ----------------------------
    # Model signature
    # ----------------------------
    signature = infer_signature(
        model_input=X_train_transformed,
        model_output=model.predict(X_train_transformed)
    )

    # ----------------------------
    # Log model
    # ----------------------------
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="classifier_pipeline",
        signature=signature,
        input_example=X_val_transformed[:1]
    )

    # ----------------------------
    # Evaluate model
    # ----------------------------
    # Include the target column in evaluation
    eval_data = pd.DataFrame(X_val_transformed, columns=X_val.columns)
    eval_data[TARGET_COL] = y_val.values

    result = mlflow.models.evaluate(
        model=model_info.model_uri,
        data=eval_data,
        targets=TARGET_COL,
        model_type="classifier",
        evaluators=["default"]
    )

    # ----------------------------
    # Print run info
    # ----------------------------
    print(f"Run ID: {run_id}")
    print(f"Validation Accuracy: {acc}")
    print(f"Validation ROC AUC: {roc}")
    print(f"Evaluation Metrics: {result.metrics}")

print("✅ Model training and logging completed")
