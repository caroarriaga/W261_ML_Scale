# Databricks notebook source
# MAGIC %md
# MAGIC #### Custom Grid Search cross validation which accounts for time series

# COMMAND ----------

# import libraries
import pyspark.sql.functions as F
from pyspark.sql.types import *
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.ml.linalg import DenseVector, SparseVector, Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator



# COMMAND ----------

spark.sparkContext.addPyFile("dbfs:/custom_cv.py")
from custom_cv import CustomCrossValidator

# COMMAND ----------

def customGridsearchCV(df_train, estimator, grid, evaluator):
    '''
    Create Time Series rolling windows cross validation folds and passes it to custom cross validator for Grid Search
    Outputs: predictions dataframe from model
    '''

    cv_input = df_train
    
    d = {}

    d['df1'] = cv_input.filter(cv_input.YEAR <= 2015)\
                       .withColumn('cv', F.when(cv_input.MONTH <= 9, 'train').otherwise('validation'))
    d['df2'] = cv_input.filter(((cv_input.YEAR == 2015) & ((cv_input.MONTH >= 10)  & (cv_input.MONTH <= 12))) | ((cv_input.YEAR==2016) & ((cv_input.MONTH >= 1)  & (cv_input.MONTH <= 9))))\
                       .withColumn('cv', F.when((cv_input.MONTH < 6) & (cv_input.MONTH > 9), 'train').otherwise('validation'))
    d['df3'] = cv_input.filter(((cv_input.YEAR == 2016) & ((cv_input.MONTH >= 7)  & (cv_input.MONTH <= 12))) | ((cv_input.YEAR==2017) & ((cv_input.MONTH >= 1)  & (cv_input.MONTH <= 6))))\
                       .withColumn('cv', F.when((cv_input.MONTH < 4) & (cv_input.MONTH > 6), 'train').otherwise('validation'))
    d['df4'] = cv_input.filter((cv_input.YEAR == 2017) & (cv_input.MONTH >= 4) | ((cv_input.YEAR==2018) & (cv_input.MONTH <= 3))) \
                       .withColumn('cv', F.when(cv_input.MONTH >= 4, 'train').otherwise('validation'))
    d['df5'] = cv_input.filter(cv_input.YEAR == 2018) \
                       .withColumn('cv', F.when(cv_input.MONTH <= 9, 'train').otherwise('validation'))
    
    cv = CustomCrossValidator(estimator=estimator, estimatorParamMaps=grid, evaluator=evaluator,splitWord = ('train', 'validation'), cvCol = 'cv', parallelism=4)

    cvModel = cv.fit(d)
    
    predictions = cvModel.transform(cv_input)
    
    predictions = predictions.withColumnRenamed('label','DEP_DEL15')
    
    return predictions
