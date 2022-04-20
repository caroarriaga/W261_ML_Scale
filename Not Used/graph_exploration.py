# Databricks notebook source
import pyspark
from pyspark.sql.types import StringType, BooleanType, IntegerType
import pyspark.sql.functions as F

import airporttime
from datetime import datetime, timedelta

import numpy as np

from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml.feature import Bucketizer
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from sparkdl.xgboost import XgboostRegressor, XgboostClassifier

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

# MAGIC %run "./libs/weather_aggregation"

# COMMAND ----------

# MAGIC %run "./libs/time_based_features"

# COMMAND ----------

# MAGIC %run "./libs/transform"

# COMMAND ----------

# MAGIC %run "./libs/model_helper_functions"

# COMMAND ----------

df_train = spark.read.parquet(f"{blob_url}/cv_train_0402_split1")
# df_test = spark.read.parquet(f"{blob_url}/cv_val_0402_split1")
df1 = spark.read.parquet(f"{blob_url}/holidays")

def add_holidays(df_train, df1):

    # # change date to correct format
    for column in ["Month", "Day"]:
        df1 = df1.withColumn(column, F.when(F.length(F.col(column))<2, F.lpad(F.col(column), 2, "0")).otherwise(F.col(column)))
    df1 = df1.withColumn("Date", F.concat(col("Year"), F.lit("-"), col("Month"), F.lit("-"), F.col("Day")))
    df1 = df1.withColumn("Date_time", F.to_timestamp(df1.Date, "yyyy-mm-dd"))
    # df1 = df1.withColumn("Is_Holiday", 1).select('Date', 'Holiday', 'Date_time', 'Importance', 'Is_Holiday')

    df_holidays_train = df_train.join(df1, df_train.FL_DATE == df1.Date, how = 'left').cache()
    
    df_holidays_train=df_holidays_train.withColumn("is_holiday", F.when(df_holidays_train.Importance >=0, 1).otherwise(0))
    df_holidays_train=df_holidays_train.withColumn("holiday_imp", F.when(df_holidays_train.Importance ==0, 1) \
                                               .when(df_holidays_train.Importance ==1, 2) \
                                               .when(df_holidays_train.Importance ==2, 3) \
                                               .otherwise(0))
    df_holidays_train=df_holidays_train.drop(*['Date','Holiday','WeekDay','Month','Day','Year','Importance','Date_time'])
    
    return df_holidays_train

df_train_with_holidays = add_holidays(df_train,df1).cache()
# display(df_train_with_holidays)


# COMMAND ----------


display(df_train_with_holidays)

# COMMAND ----------

# MAGIC %md
# MAGIC Create a graph of connected airports

# COMMAND ----------

import re
import heapq
import itertools
import numpy as np
import networkx as nx
from graphframes import *
from collections import defaultdict, deque
import matplotlib.pyplot as plt

# COMMAND ----------

vertices = df_train_with_holidays.groupBy("ORIGIN_AIRPORT_ID").agg(F.first('ORIGIN').alias('src')) \
                                 .withColumnRenamed("ORIGIN_AIRPORT_ID","id")

display(vertices)

# COMMAND ----------

edges = df_train_with_holidays.select(
                              F.col('ORIGIN').alias('src'), \
                              F.col('DEST').alias('dst'), \
                              F.col('OP_UNIQUE_CARRIER').alias('carrier'), \
                              F.col('PREV_DEP_DEL15').alias('prev_delay'), \
                              F.col('DEP_DEL15'))

edges

# COMMAND ----------



# COMMAND ----------

g = GraphFrame(vertices,edges)

# COMMAND ----------

inDegreeOrigin = g.inDegrees

# COMMAND ----------

display(inDegreeOrigin)

# COMMAND ----------

# inDegreeOrigin.write.parquet(f"{blob_url}/inDegreeOriginTable")

# COMMAND ----------

avg_route_delay = g.edges\
 .groupBy('src','dst','carrier')\
 .avg('prev_delay').withColumnRenamed('avg(prev_delay)','avg_prev_delay_carrier')

# COMMAND ----------

# avg_route_delay.write.parquet(f"{blob_url}/prevDelTable")

# COMMAND ----------

mostImportantOrigin=df_train_with_holidays.groupBy('ORIGIN')\
                              .agg(F.avg("inDegree").alias("avg_indegree"))\
                              .sort(F.desc("avg_indegree"))

display(mostImportantOrigin)


# COMMAND ----------

mostImportantOrigin.approxQuantile('avg_indegree', [0.5],.25)


# COMMAND ----------

medianInDegree

# COMMAND ----------

display(df_train_with_holidays)

# COMMAND ----------

