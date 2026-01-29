# Databricks notebook source
# notebooks/03_train_model.py

import logging
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# Configure Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------
# Environment
# ----------------------------
ENV = dbutils.widgets.get("env")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
feature_pipeline_path = dbutils.widgets.get("feature_pipeline")

logger.info(f"Environment: {ENV}")
logger.info(f"Experiment Name: {EXPERIMENT_NAME}")
logger.info(f"Feature Pipeline Path: {feature_pipeline_path}")

CATALOG = f"mlops_{ENV}"
SCHEMA = "feature_engineering"
logger.info(f"Catalog: {CATALOG}, Schema: {SCHEMA}")

# ----------------------------
# Experiment setup
# ----------------------------
logger.info(f"Setting MLflow experiment to {EXPERIMENT_NAME}...")
mlflow.set_experiment(EXPERIMENT_NAME)

# ----------------------------
# Load feature pipeline
# ----------------------------
logger.info(f"Loading feature pipeline from {feature_pipeline_path}...")
feature_pipeline = joblib.load(feature_pipeline_path)
logger.info("Feature pipeline loaded successfully")

# ----------------------------
# Load train/val data
# ----------------------------
logger.info("Loading train/val data from Delta tables...")
train_sdf = spark.table(f"{CATALOG}.{SCHEMA}.train_split")
val_sdf = spark.table(f"{CATALOG}.{SCHEMA}.val_split")

train_df = train_sdf.toPandas()
val_df = val_sdf.toPandas()
logger.info(f"Train data shape: {train_df.shape}, Val data shape: {val_df.shape}")

TARGET_COL = "Potability"
logger.info(f"Target column: {TARGET_COL}")

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_val = val_df.drop(columns=[TARGET_COL])
y_val = val_df[TARGET_COL]
logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# ----------------------------
# Transform features
# ----------------------------
logger.info("Transforming features using feature pipeline...")
X_train_transformed = feature_pipeline.transform(X_train)
X_val_transformed = feature_pipeline.transform(X_val)
logger.info(f"X_train_transformed shape: {X_train_transformed.shape}")
logger.info(f"X_val_transformed shape: {X_val_transformed.shape}")

# ----------------------------
# Train model
# ----------------------------
logger.info("Initializing LogisticRegression model...")
model = LogisticRegression(max_iter=1000, class_weight="balanced")

logger.info("Starting MLflow run for model training...")
with mlflow.start_run(run_name="logreg_pipeline") as run:
    run_id = run.info.run_id
    logger.info(f"MLflow run ID: {run_id}")

    # Fit the model
    logger.info("Fitting model on transformed training data...")
    model.fit(X_train_transformed, y_train)
    logger.info("Model fitted successfully")

    # ----------------------------
    # Predictions & Metrics
    # ----------------------------
    logger.info("Generating predictions and metrics...")
    y_pred = model.predict(X_val_transformed)
    y_prob = model.predict_proba(X_val_transformed)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_prob)
    logger.info(f"Validation Accuracy: {acc:.4f}")
    logger.info(f"Validation ROC AUC: {roc:.4f}")

    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("val_roc_auc", roc)
    logger.info("Metrics logged to MLflow")

    # ----------------------------
    # Log original datasets
    # ----------------------------
    logger.info("Logging train/val datasets to MLflow...")
    train_dataset = mlflow.data.from_spark(train_sdf, table_name=f"{CATALOG}.{SCHEMA}.train_split")
    mlflow.log_input(train_dataset, context="training")

    val_dataset = mlflow.data.from_spark(val_sdf, table_name=f"{CATALOG}.{SCHEMA}.val_split")
    mlflow.log_input(val_dataset, context="validation")
    logger.info("Datasets logged to MLflow")

    # ----------------------------
    # Model signature
    # ----------------------------
    logger.info("Inferring model signature...")
    signature = infer_signature(
        model_input=X_train_transformed,
        model_output=model.predict(X_train_transformed)
    )
    logger.info("Model signature inferred")

    # ----------------------------
    # Log model
    # ----------------------------
    logger.info("Logging model to MLflow...")
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="classifier_pipeline",
        signature=signature,
        input_example=X_val_transformed[:1]
    )
    logger.info(f"Model logged to MLflow at {model_info.model_uri}")

    # ----------------------------
    # Evaluate model
    # ----------------------------
    logger.info("Evaluating model...")
    eval_data = pd.DataFrame(X_val_transformed, columns=X_val.columns)
    eval_data[TARGET_COL] = y_val.values

    result = mlflow.models.evaluate(
        model=model_info.model_uri,
        data=eval_data,
        targets=TARGET_COL,
        model_type="classifier",
        evaluators=["default"]
    )
    logger.info(f"Evaluation Metrics: {result.metrics}")

    # ----------------------------
    # Print run info
    # ----------------------------
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Validation Accuracy: {acc:.4f}")
    logger.info(f"Validation ROC AUC: {roc:.4f}")
    logger.info(f"Evaluation Metrics: {result.metrics}")

logger.info("✅ Model training and logging completed successfully")
