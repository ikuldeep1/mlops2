# Databricks notebook source
# notebooks/00_data_ingestion.py

import logging

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
logger.info(f"Environment: {ENV}")

CATALOG = f"mlops_{ENV}"
SCHEMA = "data"
logger.info(f"Catalog: {CATALOG}, Schema: {SCHEMA}")

# ----------------------------
# Read raw data
# ----------------------------
logger.info("Reading train data...")
df_train = spark.read.csv(
    "/Volumes/mlops_data/default/customvolume/train.csv",
    header=True,
    inferSchema=True)
logger.info(f"Train data shape: {df_train.count()} rows, {len(df_train.columns)} columns")

logger.info("Reading test data...")
df_test = spark.read.csv(
    "/Volumes/mlops_data/default/customvolume/test.csv",
    header=True,
    inferSchema=True)
logger.info(f"Test data shape: {df_test.count()} rows, {len(df_test.columns)} columns")

# ----------------------------
# Create catalog + schema (DEV ONLY)
# ----------------------------
logger.info(f"Creating catalog {CATALOG} if not exists...")
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
logger.info(f"Creating schema {CATALOG}.{SCHEMA} if not exists...")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# ----------------------------
# Write Delta tables
# ----------------------------
logger.info(f"Writing train data to {CATALOG}.{SCHEMA}.train...")
df_train.write.mode("overwrite").format("delta").saveAsTable(
    f"{CATALOG}.{SCHEMA}.train"
)

logger.info(f"Writing test data to {CATALOG}.{SCHEMA}.test...")
df_test.write.mode("overwrite").format("delta").saveAsTable(
    f"{CATALOG}.{SCHEMA}.test"
)

logger.info("Data ingestion completed successfully")