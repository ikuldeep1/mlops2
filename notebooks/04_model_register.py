# Databricks notebook source
# notebooks/04_register_model.py

import logging
import mlflow
from mlflow.tracking import MlflowClient

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
ENV = dbutils.widgets.get("env")                     # dev / staging / prod
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
RUN_ID = dbutils.widgets.get("run_id")               # optional
CATALOG = f"mlops_{ENV}"
SCHEMA = "model"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.water_potability_classifier"

logger.info(f"Environment: {ENV}")
logger.info(f"Experiment: {EXPERIMENT_NAME}")
logger.info(f"Run ID (input): {RUN_ID}")
logger.info(f"Model Name: {MODEL_NAME}")

# ----------------------------
# Gating thresholds
# ----------------------------
MIN_ROC_AUC = 0.51
MIN_ACCURACY = 0.52

logger.info(
    f"Registration thresholds -> "
    f"ROC_AUC >= {MIN_ROC_AUC}, Accuracy >= {MIN_ACCURACY}"
)

# ----------------------------
# Setup MLflow
# ----------------------------
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

# ----------------------------
# Resolve run_id
# ----------------------------
if not RUN_ID:
    logger.info("No run_id provided. Selecting best run by ROC AUC...")

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_roc_auc DESC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError("❌ No runs found in experiment")

    run = runs[0]
    RUN_ID = run.info.run_id
else:
    run = client.get_run(RUN_ID)

logger.info(f"Using run_id: {RUN_ID}")

# ----------------------------
# Fetch metrics
# ----------------------------
metrics = run.data.metrics
roc_auc = metrics.get("val_roc_auc")
accuracy = metrics.get("val_accuracy")

logger.info(f"Run metrics -> ROC_AUC: {roc_auc}, Accuracy: {accuracy}")

if roc_auc is None or accuracy is None:
    raise RuntimeError("❌ Required metrics not found in run")

# ----------------------------
# Gating decision
# ----------------------------
should_register = (
    roc_auc >= MIN_ROC_AUC and
    accuracy >= MIN_ACCURACY
)

logger.info(f"Registration decision: {should_register}")

if not should_register:
    logger.warning("🚫 Model did not meet registration criteria. Exiting.")
    dbutils.notebook.exit("Model failed gating checks")

# ----------------------------
# Locate logged model artifact
# ----------------------------
MODEL_ARTIFACT_PATH = "classifier_pipeline"
MODEL_URI = f"runs:/{RUN_ID}/{MODEL_ARTIFACT_PATH}"

logger.info(f"Model URI: {MODEL_URI}")

# ----------------------------
# Register model
# ----------------------------
logger.info("Registering model in MLflow Model Registry...")

registered_model = mlflow.register_model(
    model_uri=MODEL_URI,
    name=MODEL_NAME
)

model_version = registered_model.version
logger.info(
    f"✅ Model registered successfully -> "
    f"{MODEL_NAME} (version {model_version})"
)

# ----------------------------
# Tag model version
# ----------------------------
client.set_model_version_tag(
    name=MODEL_NAME,
    version=model_version,
    key="env",
    value=ENV
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=model_version,
    key="val_roc_auc",
    value=str(roc_auc)
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=model_version,
    key="val_accuracy",
    value=str(accuracy)
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=model_version,
    key="source_run_id",
    value=RUN_ID
)

logger.info("Model version tags added")

# ----------------------------
# Optional: auto-transition (DEV only)
# ----------------------------
if ENV == "dev":
    logger.info("DEV environment detected. Transitioning model to dev...")

    client.set_registered_model_alias(
        name=MODEL_NAME,
        version=model_version,
        alias="dev",
        # archive_existing_versions=False
    )

    logger.info("Model promoted to dev")

# ----------------------------
# Summary
# ----------------------------
logger.info("🎉 Model registration workflow completed successfully")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Version: {model_version}")
logger.info(f"Stage: {'Dev' if ENV == 'dev' else 'None'}")