# Databricks notebook source
# notebooks/02_feature_engineering.py

import logging
import mlflow
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
# Load validated data
# ----------------------------
logger.info("Loading validated train data...")
df = spark.table(f"{CATALOG}.{SCHEMA}.train").toPandas()
logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

TARGET_COL = "Potability"
logger.info(f"Target column: {TARGET_COL}")

# ----------------------------
# Train-Val split
# ----------------------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logger.info(f"Train/val split: X_train={X_train.shape}, X_val={X_val.shape}, y_train={y_train.shape}, y_val={y_val.shape}")

# ----------------------------
# Feature engineering
# ----------------------------
logger.info("Identifying feature types...")
numeric_features = X.columns.tolist()
categorical_features = []  # none for this dataset
logger.info(f"Numeric features: {numeric_features}")
logger.info(f"Categorical features: {categorical_features}")

# Numeric pipeline
logger.info("Building numeric pipeline...")
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline (for future-proofing)
logger.info("Building categorical pipeline...")
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Column transformer
logger.info("Building column transformer...")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        # ("cat", categorical_pipeline, categorical_features)
    ]
)

# Fit ONLY on training data
logger.info("Fitting preprocessor on training data...")
preprocessor.fit(X_train)
logger.info("Preprocessor fitted successfully")

# ----------------------------
# Save feature pipeline
# ----------------------------
feature_pipeline_path = "/Workspace/Users/krathour13@gmail.com/dbfs_data/feature_pipeline.joblib"
logger.info(f"Saving feature pipeline to {feature_pipeline_path}...")
joblib.dump(preprocessor, feature_pipeline_path)
logger.info("Feature pipeline saved successfully")

# ----------------------------
# Save train/val splits
# ----------------------------
logger.info("Preparing train/val DataFrames for saving...")
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
logger.info(f"Train DataFrame shape: {train_df.shape}, Val DataFrame shape: {val_df.shape}")

train_spark = spark.createDataFrame(train_df)
val_spark = spark.createDataFrame(val_df)

# ----------------------------
# Save train/val DataFrames as UC tables
# ----------------------------
logger.info(f"Creating schema {CATALOG}.feature_engineering if not exists...")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.feature_engineering")

output_path = f"{CATALOG}.feature_engineering"
logger.info(f"Saving train split to {output_path}.train_split...")
train_spark.write.format("delta").mode("overwrite").saveAsTable(f"{output_path}.train_split")
logger.info(f"Saving val split to {output_path}.val_split...")
val_spark.write.format("delta").mode("overwrite").saveAsTable(f"{output_path}.val_split")

logger.info("✅ Feature pipeline trained and saved. Train/val splits saved as Delta tables.")