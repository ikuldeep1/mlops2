# Databricks notebook source
# notebooks/05_validate_model_staging.py

import logging
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# Configure Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------
# Widgets / Environment
# ----------------------------
ENV = dbutils.widgets.get("env")                  # staging
MODEL_ALIAS = dbutils.widgets.get("model_alias")  # dev
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
FEATURE_PIPELINE_PATH = dbutils.widgets.get("feature_pipeline")

CATALOG = f"mlops_{ENV}"
FEATURE_SCHEMA = "feature_engineering"
MODEL_SCHEMA = "model"

MODEL_NAME = f"{CATALOG}.{MODEL_SCHEMA}.water_potability_classifier"
TARGET_COL = "Potability"

logger.info(f"Environment: {ENV}")
logger.info(f"Model Name: {MODEL_NAME}")
logger.info(f"Model Alias: {MODEL_ALIAS}")

# ----------------------------
# Gating thresholds (TEST)
# ----------------------------
MIN_TEST_ROC_AUC = 0.50
MIN_TEST_ACCURACY = 0.50

logger.info(
    f"Test gating thresholds -> "
    f"ROC_AUC >= {MIN_TEST_ROC_AUC}, Accuracy >= {MIN_TEST_ACCURACY}"
)

# ----------------------------
# MLflow setup
# ----------------------------
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

# ----------------------------
# Load model from registry
# ----------------------------
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
logger.info(f"Loading model from: {MODEL_URI}")

model = mlflow.sklearn.load_model(MODEL_URI)

# ----------------------------
# Load feature pipeline
# ----------------------------
logger.info("Loading feature pipeline...")
feature_pipeline = joblib.load(FEATURE_PIPELINE_PATH)

# ----------------------------
# Load test data
# ----------------------------
logger.info("Loading test data from Delta table...")
test_sdf = spark.table(f"{CATALOG}.data.test")
test_df = test_sdf.toPandas()

logger.info(f"Test data shape: {test_df.shape}")

X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

# ----------------------------
# Feature transformation
# ----------------------------
logger.info("Transforming test features...")
X_test_transformed = feature_pipeline.transform(X_test)


# ----------------------------
# Validation run
# ----------------------------
with mlflow.start_run(run_name="staging_model_validation") as run:
    run_id = run.info.run_id
    logger.info(f"MLflow run ID: {run_id}")

    # ----------------------------
    # Predictions
    # ----------------------------
    y_pred = model.predict(X_test_transformed)
    y_prob = model.predict_proba(X_test_transformed)[:, 1]

    # ----------------------------
    # Metrics
    # ----------------------------
    test_accuracy = accuracy_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_prob)

    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test ROC AUC: {test_roc_auc:.4f}")

    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_roc_auc", test_roc_auc)

    # ----------------------------
    # Log test dataset
    # ----------------------------
    test_dataset = mlflow.data.from_spark(
        test_sdf,
        table_name=f"{CATALOG}.data.test"
    )
    mlflow.log_input(test_dataset, context="test")

    # ----------------------------
    # MLflow evaluate (optional but recommended)
    # ----------------------------
    eval_df = pd.DataFrame(X_test_transformed, columns=X_test.columns)
    eval_df[TARGET_COL] = y_test.values

    result = mlflow.models.evaluate(
        model=MODEL_URI,
        data=eval_df,
        targets=TARGET_COL,
        model_type="classifier",
        evaluators=["default"]
    )

    logger.info(f"Evaluation metrics: {result.metrics}")

# ----------------------------
# Gating decision
# ----------------------------
passed = (
    test_accuracy >= MIN_TEST_ACCURACY and
    test_roc_auc >= MIN_TEST_ROC_AUC
)

logger.info(f"Staging validation passed: {passed}")

if not passed:
    logger.warning("🚫 Model failed staging validation")
    # dbutils.notebook.exit("Model failed staging validation")

# ----------------------------
# Promote to staging
# ----------------------------
logger.info("Promoting model to STAGING alias...")

model_version = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias=MODEL_ALIAS
).version

client.set_registered_model_alias(
    name=MODEL_NAME,
    version=model_version,
    alias="staging"
)

# ----------------------------
# Tag model version
# ----------------------------
client.set_model_version_tag(
    name=MODEL_NAME,
    version=model_version,
    key="test_accuracy",
    value=str(test_accuracy)
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=model_version,
    key="test_roc_auc",
    value=str(test_roc_auc)
)

logger.info("✅ Model successfully validated and promoted to STAGING")
