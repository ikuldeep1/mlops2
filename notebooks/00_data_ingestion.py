# Databricks notebook source
# notebooks/00_data_ingestion.py

ENV = dbutils.widgets.get("env")
print(ENV)
# from pyspark.sql import functions as F
# from sklearn.model_selection import train_test_split

# # ----------------------------
# # Environment
# # ----------------------------
# dbutils.widgets.text("env", "dev")
# ENV = dbutils.widgets.get("env")

# CATALOG = f"mlops_{ENV}"
# SCHEMA = "raw"

# # ----------------------------
# # Read raw data
# # ----------------------------
# df = spark.read.csv(
#     "/Volumes/workspace/default/customvolume/water_potability.csv",
#     header=True,
#     inferSchema=True
# )

# columns = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']  # numeric columns only

# medians = {
#     c: df.select(F.expr(f"percentile_approx({c}, 0.5)")).first()[0]
#     for c in columns
# }

# df = df.fillna(medians)

# pdf = df.toPandas()

# # ----------------------------
# # Stratified split
# # ----------------------------
# train_pdf, temp_pdf = train_test_split(
#     pdf,
#     test_size=0.30,
#     stratify=pdf["Potability"],
#     random_state=42
# )

# val_pdf, test_pdf = train_test_split(
#     temp_pdf,
#     test_size=0.50,
#     stratify=temp_pdf["Potability"],
#     random_state=42
# )

# # ----------------------------
# # Pandas → Spark
# # ----------------------------
# train_df = spark.createDataFrame(train_pdf)
# val_df   = spark.createDataFrame(val_pdf)
# test_df  = spark.createDataFrame(test_pdf)

# # ----------------------------
# # Create catalog + schema (DEV ONLY)
# # ----------------------------
# spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
# spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# # ----------------------------
# # Write Delta tables
# # ----------------------------
# train_df.write.mode("overwrite").format("delta").saveAsTable(
#     f"{CATALOG}.{SCHEMA}.train"
# )

# val_df.write.mode("overwrite").format("delta").saveAsTable(
#     f"{CATALOG}.{SCHEMA}.val"
# )

# test_df.write.mode("overwrite").format("delta").saveAsTable(
#     f"{CATALOG}.{SCHEMA}.test"
# )

# print("DEV data ingestion completed")