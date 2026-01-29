# Databricks notebook source
# notebooks/01_data_validation.py

import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, current_timestamp, lit

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

# Set up spark session
spark = SparkSession.builder.appName("WaterPotabilityValidation").getOrCreate()
logger.info("Spark session initialized")

# ----------------------------
# Load data
# ----------------------------
logger.info("Loading train data...")
train_df = spark.table(f"{CATALOG}.{SCHEMA}.train")
logger.info(f"Train data loaded: {train_df.count()} rows, {len(train_df.columns)} columns")

# ----------------------------
# Gate 1: Non-empty dataset
# ----------------------------
row_count = train_df.count()
if row_count == 0:
    error_msg = "❌ Data validation failed: train dataset is empty"
    logger.error(error_msg)
    raise ValueError(error_msg)
logger.info("✅ Gate 1 passed: Dataset is not empty")

# ----------------------------
# Step 3: Check for null values per column
# ----------------------------
logger.info("Checking for null values per column...")
null_counts = train_df.select([count(when(col(c).isNull(), c)).alias(c) for c in train_df.columns])
logger.info("Null values per column:\n" + "\n".join([f"{row}" for row in null_counts.collect()]))

# ----------------------------
# Step 4: Check for duplicates
# ----------------------------
dup_count = train_df.count() - train_df.dropDuplicates().count()
logger.info(f"Number of duplicate rows: {dup_count}")

# ----------------------------
# Step 5: Validate column types and constraints
# ----------------------------
logger.info("Validating column constraints...")
validation_errors = []

# pH should be between 0 and 14
invalid_ph = train_df.filter((col("ph") < 0) | (col("ph") > 14)).count()
if invalid_ph > 0:
    validation_errors.append(f"Invalid pH values: {invalid_ph}")

# Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity should be non-negative
numeric_cols = ["Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]
for c in numeric_cols:
    invalid_count = train_df.filter(col(c) < 0).count()
    if invalid_count > 0:
        validation_errors.append(f"Negative values in {c}: {invalid_count}")

# Potability should be 0 or 1
invalid_potability = train_df.filter(~col("Potability").isin(0,1)).count()
if invalid_potability > 0:
    validation_errors.append(f"Invalid Potability values: {invalid_potability}")

# ----------------------------
# Step 6: Report results
# ----------------------------
if len(validation_errors) == 0:
    logger.info("✅ Data validation passed. No critical issues found.")
else:
    logger.warning("⚠️ Data validation failed with the following issues:")
    for err in validation_errors:
        logger.warning(f"- {err}")

# ----------------------------
# Gate 2: Required columns
# ----------------------------
required_columns = {"Potability"}
missing_cols = required_columns - set(train_df.columns)
if missing_cols:
    error_msg = f"❌ Missing required columns: {missing_cols}"
    logger.error(error_msg)
    raise ValueError(error_msg)
logger.info("✅ Gate 2 passed: All required columns are present")

# ----------------------------
# Gate 3: Target validity
# ----------------------------
class_counts = train_df.groupBy("Potability").count().collect()
if len(class_counts) < 2:
    error_msg = "❌ Only one class present in training data"
    logger.error(error_msg)
    raise ValueError(error_msg)
logger.info("✅ Gate 3 passed: Both classes are present in the target")

# ----------------------------
# Gate 4: Null ratio
# ----------------------------
total_rows = train_df.count()
null_ratio_errors = []
for column in train_df.columns:
    nulls = train_df.filter(train_df[column].isNull()).count()
    if nulls / total_rows > 0.1:
        null_ratio_errors.append(f"Column {column} has >10% nulls: {nulls}/{total_rows}")
if null_ratio_errors:
    error_msg = "\n".join(null_ratio_errors)
    logger.error(f"❌ Null ratio validation failed:\n{error_msg}")
    raise ValueError(error_msg)
logger.info("✅ Gate 4 passed: No column has >10% nulls")

# ----------------------------
# Step 7: Save validation report to Delta
# ----------------------------
logger.info("Saving validation report...")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.validation_report")

report_path = f"{CATALOG}.validation_report.water_potability_validation"

# Prepare validation summary
validation_summary = spark.createDataFrame(
    [
        (
            str([row.asDict() for row in null_counts.collect()]),
            dup_count,
            str(validation_errors),
            str(null_ratio_errors),
            str([row.asDict() for row in class_counts])
        )
    ],
    [
        "null_counts",
        "duplicates",
        "constraint_errors",
        "null_ratio_errors",
        "class_distribution"
    ]
)
validation_summary = validation_summary.withColumn("validation_timestamp", current_timestamp())

# Save report
validation_summary.write.format("delta").mode("append").saveAsTable(report_path)
logger.info(f"✅ Validation report saved to {report_path}")