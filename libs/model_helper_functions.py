# Databricks notebook source
import pyspark
from pyspark.sql.types import StringType, BooleanType, IntegerType
import pyspark.sql.functions as F
from itertools import chain

import airporttime
from datetime import datetime, timedelta

import numpy as np
import random as rd

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
from sparkdl.xgboost import XgboostRegressor
import catboost_spark

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Cross validation & model functions

# COMMAND ----------

def getRegressionPipeline(categoricalColumns, numericCols, labelCol):
  
  '''
  Build a ML pipeline for Logistic Regression 
    Bucketizes CRS_DEP_TIME
    Creates one hot encoding for categorical columns
    Adds Vectorization
    Adds Scaling
  Outputs: Pyspark ML pipeline object
  '''
  
  stages = []

  # use bucketizer to create buckets
  # After creating buckets, it'll be treated as categorical column
  if "CRS_DEP_TIME" in numericCols:
    bucketizer = Bucketizer(splits=[ 0, 900, 1200, 1600, 2000, 2359],inputCol="CRS_DEP_TIME", outputCol="CRS_DEP_BUCKET", handleInvalid = "keep")
    stages += [bucketizer]
    numericCols.remove("CRS_DEP_TIME")

  # for categorical columns, do one hot encoding
  for categoricalCol in categoricalColumns:
      stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + '_Index', handleInvalid="keep")
      encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "_classVec"],handleInvalid = "keep")
      stages += [stringIndexer, encoder]

  assemblerInputs = [c + "_classVec" for c in categoricalColumns] + numericCols

  assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="vectorized_features",handleInvalid = "keep")
  stages += [assembler]

  # scale the features
  scaler = StandardScaler(inputCol="vectorized_features", outputCol="features")
  stages += [scaler]

  #create pipeline of all the tranformations needed
  pipeline = Pipeline(stages = stages)
    
  return pipeline


# COMMAND ----------

def getXGBPipeline(numericCols):
  
  '''
  Build a ML pipeline for XGBoost 
    Bucketizes CRS_DEP_TIME
    Adds Vectorization
  Outputs: Pyspark ML pipeline object
  '''
  
  stages = []

  # use bucketizer to create buckets
  # After creating buckets, it'll be treated as categorical column
  if "CRS_DEP_TIME" in numericCols:
    bucketizer = Bucketizer(splits=[ 0, 900, 1200, 1600, 2000, 2359],inputCol="CRS_DEP_TIME", outputCol="CRS_DEP_BUCKET", handleInvalid = "keep")
    stages += [bucketizer]
    numericCols.remove("CRS_DEP_TIME")

  # for categorical columns, do one hot encoding
#   for categoricalCol in categoricalColumns:
#       stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + '_Index', handleInvalid="keep")
#   encoder = OneHotEncoder(inputCols=['weather_condition_Index'], outputCols=['weather_condition_vec'],  handleInvalid = "keep")
#   stages += [encoder]

#   assemblerInputs = numericCols + ['weather_condition_vec']
#   assemblerInputs.remove('weather_condition_Index')

  assembler = VectorAssembler(inputCols=numericCols, outputCol="features",handleInvalid = "keep")
  stages += [assembler]

  # scale the features
#   scaler = StandardScaler(inputCol="vectorized_features", outputCol="features")
#   stages += [scaler]

  #create pipeline of all the tranformations needed
  pipeline = Pipeline(stages = stages)
    
  return pipeline

# COMMAND ----------

def Split4year5Fold(df):
  
  '''
  Creates 5 splits on a time series dataframe
    Each train split contains 9 months of data
    Each validation split contains 3 months of data
  Outputs: Five train & validation dataframes
  '''
    
  trainsplits = []
  testsplits =[]
  
  train1 = df.filter((df.YEAR=="2015") & ((df.MONTH >= 1)  & (df.MONTH <= 9)))
  test1 = df.filter((df.YEAR=="2015") & ((df.MONTH >= 10)  & (df.MONTH <= 12)))
  
  train2 = df.filter(((df.YEAR=="2015") & ((df.MONTH >= 10)  & (df.MONTH <= 12))) | ((df.YEAR=="2016") & ((df.MONTH >= 1)  & (df.MONTH <= 6))))
  test2 = df.filter((df.YEAR=="2016") & ((df.MONTH >= 7)  & (df.MONTH <= 9)))
  
  train3 = df.filter(((df.YEAR=="2016") & ((df.MONTH >= 7)  & (df.MONTH <= 12))) | ((df.YEAR=="2017") & ((df.MONTH >= 1)  & (df.MONTH <= 3))))
  test3 = df.filter((df.YEAR=="2017") & ((df.MONTH >= 4)  & (df.MONTH <= 6)))

  train4 = df.filter((df.YEAR=="2017") & ((df.MONTH >= 4)  & (df.MONTH <= 12)))
  test4 = df.filter((df.YEAR=="2018") & ((df.MONTH >= 1)  & (df.MONTH <= 3)))
  
  train5 = df.filter((df.YEAR=="2018") & ((df.MONTH >= 1)  & (df.MONTH <= 9)))
  test5 = df.filter((df.YEAR=="2018") & ((df.MONTH >= 10)  & (df.MONTH <= 12)))
  
  trainsplits.append(train1)
  trainsplits.append(train2)
  trainsplits.append(train3)
  trainsplits.append(train4)
  trainsplits.append(train5)
  testsplits.append(test1)
  testsplits.append(test2)
  testsplits.append(test3)
  testsplits.append(test4)
  testsplits.append(test5)
  
  return trainsplits, testsplits
  

# COMMAND ----------

def execLinearModel(train, test, regVal=0.0, cutoff=0.5, iter=10):
  
  '''
  Creates Logistic Regression model on training set and does prediction on test set with fitted model
  Outputs: model & prediction
  '''
  
  lr = LogisticRegression(featuresCol = 'features', labelCol = 'DEP_DEL15', regParam=regVal, threshold=cutoff, maxIter=iter)
  lrModel = lr.fit(train)
  predictions = lrModel.transform(test)
  #predictions_train = lrModel.transform(train)
  #display(predictions.select('DEP_DEL15', 'features',  'rawPrediction', 'prediction', 'probability'))
  
  return lrModel, predictions

# COMMAND ----------

def execRFModel(train, test, maxDepth=5, numTrees=20):

  '''
  Creates Random Forest model on training set and does prediction on test set with fitted model
  Outputs: predictions dataframe
  '''
    
  rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'DEP_DEL15', maxDepth=maxDepth, numTrees=numTrees)
  rfModel = rf.fit(train)
  predictions = rfModel.transform(test)
  #display(predictions.select('DEP_DEL15', 'features',  'rawPrediction', 'prediction', 'probability'))

  return predictions

# COMMAND ----------

def execXGBModel(train,test):
  
  '''
  Creates XGBoost regression model on training set and does prediction on test set with fitted model
  Outputs: predictions dataframe
  '''

  rf = XgboostRegressor(featuresCol = 'features', 
                        labelCol = 'DEP_DEL15',
                        missing=0.0,
                        colsample_bytree=0.5)
  rfModel = rf.fit(train)
  predictions = rfModel.transform(test)
  #display(predictions.select('DEP_DEL15', 'features',  'rawPrediction', 'prediction', 'probability'))

  return predictions

# COMMAND ----------

def execXGBModelClass(train,test, params):

  '''
  Creates XGBoost classification model on training set and does prediction on test set with fitted model
  Outputs: model & predictions dataframe
  '''
  
  xgb = XgboostClassifier(featuresCol = 'features', 
                         labelCol = 'DEP_DEL15',
                         max_depth = params[0],
                         n_estimators = params[1],
                         learning_rate = params[2],
                         gamma = params[3],
                         reg_alpha = params[4],
                         reg_lambda = params[5],
                         scale_pos_weight=params[6],
                         missing=0.0,
                         random_state=42,
                         colsample_bytree=0.5)
  xgbModel = xgb.fit(train)
  predictions = xgbModel.transform(test)
  #display(predictions.select('DEP_DEL15', 'features',  'rawPrediction', 'prediction', 'probability'))

  return predictions, xgbModel

# COMMAND ----------

def execXGBModelClass_custom(train,test,params):
  
  '''
  Creates XGBoost classification model on training set and does prediction on test set with fitted model
  Outputs: model & predictions dataframe
  '''
  
  
  rf = XgboostClassifier(featuresCol = 'features', 
                         labelCol = 'DEP_DEL15',
                         max_depth = params[0],
                         n_estimators = params[1],
                         learning_rate = params[2],
                         gamma = params[3],
                         reg_alpha = params[4],
                         reg_lambda = params[5],
                         missing=0.0,
                         random_state=42,
                         colsample_bytree=0.5
                        )
  xgbModel = rf.fit(train)
  predictions = xgbModel.transform(test)
  #display(predictions.select('DEP_DEL15', 'features',  'rawPrediction', 'prediction', 'probability'))

  return predictions, xgbModel

# COMMAND ----------

def execXGBModelClass_default(train,test):

  '''
  Creates XGBoost classification model on training set and does prediction on test set with fitted model
  Outputs: model & predictions dataframe
  '''
    
  rf = XgboostClassifier(featuresCol = 'features', 
                         labelCol = 'DEP_DEL15',
                         missing=0.0,
                         random_state=42)
  xgbModel = rf.fit(train)
  predictions = xgbModel.transform(test)
  #display(predictions.select('DEP_DEL15', 'features',  'rawPrediction', 'prediction', 'probability'))

  return predictions, xgbModel

# COMMAND ----------

def execCatboostModel(train, test, hyper_params = None):
  """
  Trains a CatBoost Classifier on the training data set and then generates predictions for the test data set
  Inputs:
  - train: Spark DataFrame of the training data
  - test: Spark DataFrame of the test data
  - hyper_params: Dictionary of hyper parameter name and value
  Outputs:
  - model: the trained model
  - predictions: predictions generated from the test data set 
  """
  train_pool = catboost_spark.Pool(train).setLabelCol("DEP_DEL15")
  test_pool = catboost_spark.Pool(test).setLabelCol("DEP_DEL15")
  cb_classifier = catboost_spark.CatBoostClassifier(**hyper_params, labelCol = "DEP_DEL15")
  model = cb_classifier.fit(train_pool)
  predictions = model.transform(test_pool.data)
  return model, predictions

# COMMAND ----------

def getMetrics(predictions):

  '''
  Takes in a prediction dataframe and calculates ML metrics
  Outputs: precision, recall and F measure(0.5) 
  '''
    
  TN = predictions.filter('prediction == 0 AND DEP_DEL15 == 0').count()
  TP = predictions.filter('prediction == 1 AND DEP_DEL15 == 1').count()
  FN = predictions.filter('prediction == 0 AND DEP_DEL15 != 0').count()
  FP = predictions.filter('prediction == 1 AND DEP_DEL15 != 1').count()
  
  if (TP + FP) > 0:
    precision = TP / (TP + FP)
  else:
    precision = 0
    
  if (TP + FN) > 0:
    recall = TP / (TP + FN)
  else:
    recall=0
  
  beta = 0.5

  if (precision + recall) > 0:
    fmeasure = ((1 + (beta ** 2))  * precision*recall)/ (((beta ** 2) * precision) + recall)
  else:
    fmeasure = 0
  
  return precision, recall, fmeasure

# COMMAND ----------

# function target encoding for features with too many indices
def target_mean_encoding(df, col, target):
    """
    :param df: pyspark.sql.dataframe
        dataframe to apply target mean encoding
    :param col: str list
        list of columns to apply target encoding
    :param target: str
        target column
    :return:
        dataframe with target encoded columns added
    """

    for c in col:
      
        means = df.groupby(F.col(c)).agg(F.mean(target).alias("mean_encoding"))
        
        mean_dict = {row[c]: row['mean_encoding'] for row in means.collect()}
        
        mapping_expr = F.create_map([F.lit(x) for x in chain(*mean_dict.items())])

        df = df.withColumn(c+"_mean_encoding", mapping_expr[F.col(c)])

    return df

# COMMAND ----------

def oversampling(df):
  
    '''
    Takes in a dataframe and oversamples the minority class so that counts of majority & minority rows are as close as possible
    Output: Dataframe with same columns as input but more rows
    '''

    minor = df.where(df.DEP_DEL15 == 1)
    count_delay = minor.count()

    major = df.where(df.DEP_DEL15 == 0)
    count_non_delay = major.count()

    mult = int(count_non_delay / count_delay)

    over_df = minor.withColumn("holder", F.explode(F.array([F.lit(x) for x in range(mult)]))).drop('holder')

    combined_df = major.unionAll(over_df)
    
    return combined_df.cache()

# COMMAND ----------

def undersampling(df):

    '''
    Takes in a dataframe and undersamples the majority class so that counts of majority & minority rows are as close as possible
    Output: Dataframe with same columns as input but less rows
    '''

    minor = df.where(df.DEP_DEL15 == 1)
    count_delay = minor.count()

    major = df.where(df.DEP_DEL15 == 0)
    count_non_delay = major.count()

    ratio = count_non_delay / count_delay

    sample_df = major.sample(False, 1/ratio)

    combined_df = minor.unionAll(sample_df)
    return combined_df.cache()

# COMMAND ----------

def get_parameters():

  '''
  Get parameters for XGBoost
  Output: parameter list
  '''
  parameters = {
    'max_depth' : [3,4,6],
    'n_estimators' : [20,50,70],
    'learning_rate' : [0.1, 0.15],
    'gamma': [0,1,5],
    'reg_alpha': [0.1, 0.2, 0.5],
    'reg_lambda': [1,5,8],
    'scale_pos_weight':[0.5, 1.0, 2.0]
  }
  
  params = [rd.choice(parameters['max_depth']),
                rd.choice(parameters['n_estimators']),
                rd.choice(parameters['learning_rate']),
                rd.choice(parameters['gamma']),
                rd.choice(parameters['reg_alpha']),
                rd.choice(parameters['reg_lambda']),
                rd.choice(parameters['scale_pos_weight'])]
  
  return params
  

# COMMAND ----------

def cv_to_skip(n):
  
  '''
  Chooses folds to skip to reduce number of folds for CV
  Output: list of numbers
  '''
  
  options=list(range(n))
  skip=rd.choice(options)
  options.remove(skip)
  skipTwo=rd.choice(options)
  
  return [skip,skipTwo]

# COMMAND ----------

def getMetricsEnsemble(predictions, hard = 1):
  
  '''
  Takes in a prediction dataframe and calculates ML metrics. Used for ensemble classifier.
    Hard vote: Based on predictions of dataframes
    Soft vote: Based on probabilities of dataframes
  Outputs: precision, recall and F measure(0.5) 
  '''
  
    if hard: 
        TN = predictions.filter('hard_vote == 0 AND DEP_DEL15 == 0').count()
        TP = predictions.filter('hard_vote == 1 AND DEP_DEL15 == 1').count()
        FN = predictions.filter('hard_vote == 0 AND DEP_DEL15 != 0').count()
        FP = predictions.filter('hard_vote == 1 AND DEP_DEL15 != 1').count()
    else:
        TN = predictions.filter('soft_vote == 0 AND DEP_DEL15 == 0').count()
        TP = predictions.filter('soft_vote == 1 AND DEP_DEL15 == 1').count()
        FN = predictions.filter('soft_vote == 0 AND DEP_DEL15 != 0').count()
        FP = predictions.filter('soft_vote == 1 AND DEP_DEL15 != 1').count()
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    else:
        recall=0
    beta = 0.5
    if (precision + recall) > 0:
        fmeasure = ((1 + (beta ** 2))  * precision*recall)/ (((beta ** 2) * precision) + recall)
    else:
        fmeasure = 0
    return precision, recall, fmeasure