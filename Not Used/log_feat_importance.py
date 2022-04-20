# Databricks notebook source
import pyspark
from pyspark.sql.types import StringType, BooleanType, IntegerType
import pyspark.sql.functions as F

import airporttime
from datetime import datetime, timedelta

import numpy as np

# COMMAND ----------

from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml.feature import Bucketizer
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Create the Azure BLOB storage to store data for quick access when datasets are huge

# COMMAND ----------

blob_container = "w261-scrr" # The name of your container created in https://portal.azure.com
storage_account = "midsw261rv" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261scrr" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261scrrkey" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# MAGIC %run "../libs/weather_aggregation"

# COMMAND ----------

# MAGIC %run "../libs/time_based_features"

# COMMAND ----------

# MAGIC %run "../libs/transform"

# COMMAND ----------

# MAGIC %run "../libs/model_helper_functions"

# COMMAND ----------

# MAGIC %run "../libs/custom_cv"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import joined data

# COMMAND ----------

# df_train = spark.read.parquet(f"{blob_url}/join_full_0329")

# COMMAND ----------

# df_test = spark.read.parquet(f"{blob_url}/test_full_join_0404")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross Validation

# COMMAND ----------

# Transform the data and save it - run this once

# trainsplits, valsplits = Split4year5Fold(df_train)

# for i, val_train in enumerate(trainsplits):
  
#   df_train_split = aggregate_weather_reports(val_train)
#   df_val_split = aggregate_weather_reports(valsplits[i])
  
#   df_train_split = get_transformed_df(df_train_split)
#   df_val_split = get_transformed_df(df_val_split)
  
#   df_train_split = add_previous_flight_delay_indicator(df_train_split)
#   df_val_split = add_previous_flight_delay_indicator(df_val_split)
  
#   df_train_split.write.parquet(f"{blob_url}/cv_train_0407_split"+str(i))
#   df_val_split.write.parquet(f"{blob_url}/cv_val_0407_split"+str(i))
  
  
  

# COMMAND ----------

# This would be part of main flow

df_train_split = []
df_val_split = []

for i in range(1):
  
  cv_train_str = "cv_train_0407_split" + str(i)
  cv_val_str = "cv_val_0407_split" + str(i)
  
  df_train_split.append(spark.read.parquet(f"{blob_url}/{cv_train_str}"))
  df_val_split.append(spark.read.parquet(f"{blob_url}/{cv_val_str}"))



# COMMAND ----------

def preprocess(df):

  df = df.fillna(999999, subset=['CIG_CeilingHeightDim_median', 'VIS_Horizontal_median' ])
  df = df.fillna(0, subset=['AA_RainDepth','AA_RainDuration', 'AL_SnowAccumDuration_mean', 'AL_SnowAccumDepth', 'AJ1_SnowDepth_mean', 'AJ1_SnowEqWaterDepth','WND_Speed_mean', 'SLP_Value_mean'])

  
  df = df.withColumn("ORIGIN_DEST_COMBO", F.concat(col("ORIGIN"),F.lit('-'),col("DEST")))
  
  df = target_mean_encoding(df, col=['ORIGIN', 'DEST','ORIGIN_DEST_COMBO'], target='DEP_DEL15')

  df = df.withColumn("CRS_DEP_TIME",(F.regexp_replace(col("CRS_DEP_TIME"), "[:]","")).cast(IntegerType())) \
                          .withColumn("DAY_OF_WEEK",col("DAY_OF_WEEK").cast(StringType())) \
                          .withColumn("MONTH",col("MONTH").cast(StringType())) \
                          .drop('ORIGIN', 'DEST', 'ORIGIN_DEST_COMBO')

  return df

# COMMAND ----------

# test with hubs

# def add_hubs(df):
  
#   df = df.withColumn("UA_HUB_ORIG", F.when((df.OP_UNIQUE_CARRIER == "UA") & ((df.ORIGIN == "ORD") | (df.ORIGIN == "DEN") | (df.ORIGIN == "IAH") | (df.ORIGIN == "LAX") | \
#                                                    (df.ORIGIN == "EWR") | (df.ORIGIN == "SFO") | (df.ORIGIN == "IAD")), 1)
#                                      .otherwise(0))
  
#   df = df.withColumn("DL_HUB_ORIG", F.when((df.OP_UNIQUE_CARRIER == "DL") & ((df.ORIGIN == "ATL") | (df.ORIGIN == "BOS") | (df.ORIGIN == "DTW") | (df.ORIGIN == "LAX") | \
#                                                    (df.ORIGIN == "MSP") | (df.ORIGIN == "JFK") | (df.ORIGIN == "LGA") | (df.ORIGIN == "SLC") | (df.ORIGIN == "SEA")), 1)
#                                      .otherwise(0))

#   df = df.withColumn("AA_HUB_ORIG", F.when((df.OP_UNIQUE_CARRIER == "AA") & ((df.ORIGIN == "DFW") | (df.ORIGIN == "CLT") | (df.ORIGIN == "ORD") | (df.ORIGIN == "LAX") | \
#                                                    (df.ORIGIN == "MIA") | (df.ORIGIN == "JFK") | (df.ORIGIN == "LGA") | (df.ORIGIN == "PHL") | (df.ORIGIN == "PHX") | (df.ORIGIN == "DCA")), 1)
#                                      .otherwise(0))
  
#   df = df.withColumn("WN_HUB_ORIG", F.when((df.OP_UNIQUE_CARRIER == "WN") & ((df.ORIGIN == "ATL") | (df.ORIGIN == "BWI") | (df.ORIGIN == "MDW") | (df.ORIGIN == "DAL") | \
#                                                    (df.ORIGIN == "DEN") | (df.ORIGIN == "HOU") | (df.ORIGIN == "LAS") | (df.ORIGIN == "LAX") | (df.ORIGIN == "OAK") | (df.ORIGIN == "MCO") | (df.ORIGIN == "PHX")), 1)
#                                      .otherwise(0))
  
#   df = df.withColumn("AS_HUB_ORIG", F.when((df.OP_UNIQUE_CARRIER == "AS") & ((df.ORIGIN == "SEA") | (df.ORIGIN == "ANC") | (df.ORIGIN == "LAX") | (df.ORIGIN == "PDX") | \
#                                                    (df.ORIGIN == "SFO")), 1)
#                                      .otherwise(0))

#   return df




# COMMAND ----------

# select the columns we'll be using for training. This is so that we can choose columns for model and record scores.


# flights + weather + time based attribute
selected_cols = ['DEP_DEL15', 'CRS_DEP_TIME', 'OP_UNIQUE_CARRIER', 'DAY_OF_WEEK', 'DISTANCE', 'DISTANCE_GROUP', 'MONTH', 'ORIGIN', 'DEST', \
                  'CIG_CeilingHeightDim_median', 'VIS_Horizontal_median', 'AA_RainDepth','AA_RainDuration', 'AL_SnowAccumDuration_mean', \
                  'AL_SnowAccumDepth', 'AJ1_SnowDepth_mean', 'AJ1_SnowEqWaterDepth','WND_Speed_mean', 'SLP_Value_mean', \
                  'OP_CARRIER_FL_NUM', 'TAIL_NUM', 'TIMESTAMP_UTC', \
                  'PREV_DEP_DEL15']

df_temp = df_train_split[0].select(*selected_cols)

df_temp = preprocess(df_temp)

labelCol = ['DEP_DEL15']

categoricalColumns = [t[0] for t in df_temp.dtypes if t[1] =='string']
categoricalColumns.remove('OP_CARRIER_FL_NUM') # not needed for features
categoricalColumns.remove('TAIL_NUM')

numericCols = [t[0] for t in df_temp.dtypes if t[1] !='string']

numericCols.remove(*labelCol)
numericCols.remove('TIMESTAMP_UTC') # not needed for features

# COMMAND ----------

display(df_temp)

# COMMAND ----------



metricsArray = np.empty((0,3), int)

for i, cv_train in enumerate(df_train_split):
  
  cv_train = cv_train.select(*selected_cols)
  cv_val = df_val_split[i].select(*selected_cols)
  
  cv_train = preprocess(cv_train)
  cv_val = preprocess(cv_val)

  # oversampling
  # cv_train = undersampling(cv_train)
  
  pipeline = getRegressionPipeline(categoricalColumns, numericCols, labelCol)
   
  pipelineModel = pipeline.fit(cv_train)  

  val_ml_train = pipelineModel.transform(cv_train)
  val_ml_test = pipelineModel.transform(cv_val)
  
  cols = cv_train.columns
  selectedCols = ['features'] + cols
  
  train = val_ml_train.select(selectedCols)
  test = val_ml_test.select(selectedCols)
  
  print("############################")
  print("Validation Set {:d}".format(i+1))
  print("Training Dataset Count: " + str(train.count()))
  print("Test Dataset Count: " + str(test.count()))
  
  model, pred = execLinearModel(train, test, iter=20)
  
  precision, recall, fmeasure = getMetrics(pred)
  
  print("Precision is {:.3f}".format(precision))
  print("Recall is {:.3f}".format(recall))
  print("F beta(0.5) score is {:.3f}".format(fmeasure))
  
  newrow = np.array([precision, recall, fmeasure])

  metricsArray = np.append(metricsArray, [newrow], axis=0)


avgArray = np.mean(metricsArray, axis=0)

print("############################")
print("Average of Cross validation")
print("Average Precision is {:.3f}".format(avgArray[0]))
print("Average Recall is {:.3f}".format(avgArray[1]))
print("Average F beta(0.5) score is {:.3f}".format(avgArray[2])) 

  

# COMMAND ----------

print(selectedCols)
print(model.coefficients)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Run the model on test data

# COMMAND ----------

# Transform the training & test data and save it - run this once
  
# df_train_upd = aggregate_weather_reports(df_train)
# df_test_upd = aggregate_weather_reports(df_test)
  
# df_train_upd = get_transformed_df(df_train_upd)
# df_test_upd = get_transformed_df(df_test_upd)
  
# df_train_upd = add_previous_flight_delay_indicator(df_train_upd)
# df_test_upd = add_previous_flight_delay_indicator(df_test_upd)
  
# df_train_upd.write.parquet(f"{blob_url}/train_agg_0404")
# df_test_upd.write.parquet(f"{blob_url}/test_agg_0404")

# COMMAND ----------

# read the dataframes for inference - this will be part of main loop

df_train_main = spark.read.parquet(f"{blob_url}/train_agg_0404")
df_test_main = spark.read.parquet(f"{blob_url}/test_agg_0404")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Custom cross validation

# COMMAND ----------

# cv_train = df_train_main.select(*selected_cols)

# cv_train = preprocess_dos(cv_train)

# pipeline = getRegressionPipeline(categoricalColumns, numericCols, labelCol)

# pipelineModel = pipeline.fit(cv_train) 

# val_ml_train = pipelineModel.transform(cv_train)

# val_ml_train = val_ml_train.withColumn("MONTH", val_ml_train.MONTH.cast(IntegerType()))
# val_ml_train = val_ml_train.withColumn("YEAR", val_ml_train.YEAR.cast(IntegerType()))


# cols = cv_train.columns
# selectedCols = ['features'] + cols
  
# train = val_ml_train.select(selectedCols)
# train = train.withColumnRenamed('DEP_DEL15', 'label')
  
# lr = LogisticRegression(labelCol="label", featuresCol="features")

# grid = ParamGridBuilder()\
#             .addGrid(lr.regParam, [0.1,1,10])\
#             .addGrid(lr.maxIter, [5,10,20])\
#             .build()

# evaluator = BinaryClassificationEvaluator()

# predictions = customGridsearchCV(train, estimator=lr, grid=grid, evaluator=evaluator)

# display(predictions)

# COMMAND ----------

def preprocess_dos(df):

  df = df.fillna(999999, subset=['CIG_CeilingHeightDim_median', 'VIS_Horizontal_median' ])
  df = df.fillna(0, subset=['AA_RainDepth','AA_RainDuration', 'AL_SnowAccumDuration_mean', 'AL_SnowAccumDepth', 'AJ1_SnowDepth_mean', 'AJ1_SnowEqWaterDepth','WND_Speed_mean', 'SLP_Value_mean'])
  
  df = df.withColumn("ORIGIN_DEST_COMBO", F.concat(col("ORIGIN"),F.lit('-'),col("DEST")))
  
  df = target_mean_encoding(df, col=['ORIGIN', 'DEST','ORIGIN_DEST_COMBO'], target='DEP_DEL15')

  df = df.withColumn("CRS_DEP_TIME",(F.regexp_replace(col("CRS_DEP_TIME"), "[:]","")).cast(IntegerType())) \
                          .withColumn("DAY_OF_WEEK",col("DAY_OF_WEEK").cast(StringType())) \
                          .withColumn("MONTH",col("MONTH").cast(StringType())) \
                          .drop('ORIGIN', 'DEST', 'ORIGIN_DEST_COMBO')

  return df

# COMMAND ----------

# flights + weather + time based attribute
selected_cols = ['DEP_DEL15', 'CRS_DEP_TIME','OP_UNIQUE_CARRIER', 'DAY_OF_WEEK', 'DISTANCE', 'DISTANCE_GROUP', 'MONTH', 'YEAR', 'ORIGIN', 'DEST', \
                  'CIG_CeilingHeightDim_median', 'VIS_Horizontal_median', 'AA_RainDepth','AA_RainDuration', 'AL_SnowAccumDuration_mean', \
                  'AL_SnowAccumDepth', 'AJ1_SnowDepth_mean', 'AJ1_SnowEqWaterDepth','WND_Speed_mean', 'SLP_Value_mean', \
                  'OP_CARRIER_FL_NUM', 'TAIL_NUM', 'TIMESTAMP_UTC', \
                  'PREV_DEP_DEL15']

df_temp2 = df_train_main.select(*selected_cols)

df_temp2 = preprocess_dos(df_temp2)

# Get numerical, categorical values and label ready for pipeline
labelCol = ['DEP_DEL15']

categoricalColumns = [t[0] for t in df_temp2.dtypes if t[1] =='string']
categoricalColumns.remove('OP_CARRIER_FL_NUM') # not needed for features
categoricalColumns.remove('TAIL_NUM')

numericCols = [t[0] for t in df_temp2.dtypes if t[1] !='string']

numericCols.remove(*labelCol)
numericCols.remove('TIMESTAMP_UTC') # not needed for features

# COMMAND ----------

df_train_main = df_train_main.select(*selected_cols)
df_test_main = df_test_main.select(*selected_cols)

df_train_main = preprocess_dos(df_train_main)
df_test_main = preprocess_dos(df_test_main)
  
#oversampling
# df_train_main = undersampling(df_train_main)
  
pipeline = getRegressionPipeline(categoricalColumns, numericCols, labelCol)
   
pipelineModel = pipeline.fit(df_train_main)  

ml_train = pipelineModel.transform(df_train_main)
ml_test = pipelineModel.transform(df_test_main)

cols = df_train_main.columns
selectedCols = ['features'] + cols
  
train_all = ml_train.select(selectedCols)
test_all = ml_test.select(selectedCols)

print("############################")

model, pred = execLinearModel(train_all, test_all, iter=20)

precision, recall, fmeasure = getMetrics(pred)

print("Final test scores")
print("Precision is {:.3f}".format(precision))
print("Recall is {:.3f}".format(recall))
print("F beta(0.5) score is {:.3f}".format(fmeasure))

# COMMAND ----------

pred.write.parquet(f"{blob_url}/lr_test_0410")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Analyze errors

# COMMAND ----------

# MAGIC %run "./libs/error_analysis"

# COMMAND ----------

analyze_errors(pred)