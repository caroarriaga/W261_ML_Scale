# Databricks notebook source
import pyspark
from pyspark.sql.functions import col, concat, lit, regexp_replace, when, length, lpad, to_timestamp, max, mean, stddev
from pyspark.sql.types import StringType, BooleanType
import pyspark.sql.functions as F

# OHE Encoding, Scaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array

#Take random samples with replacement, build trees for each one and average.  
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# COMMAND ----------

from sparkdl.xgboost import XgboostRegressor

# COMMAND ----------

# Access to write in the team's storage
blob_container = "w261-scrr" # The name of your container created in https://portal.azure.com
storage_account = "midsw261rv" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261scrr" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261scrrkey" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Getting external data

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
display(df_airlines)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

df_joined = add_previous_flight_delay_indicator(df_joined)

# COMMAND ----------

selected_cols = ["ACTUAL_ELAPSED_TIME","AIR_TIME","ARR_DEL15","ARR_DELAY","ARR_DELAY_GROUP","ARR_DELAY_NEW","ARR_TIME",
                 "ARR_TIME_BLK","CANCELLATION_CODE","CANCELLED","CARRIER_DELAY","CRS_ARR_TIME","CRS_DEP_TIME",
                 "CRS_ELAPSED_TIME","DAY_OF_MONTH","DAY_OF_WEEK","DEP_DEL15","DEP_DELAY","DEP_DELAY_GROUP",
                 "DEP_DELAY_NEW","DEP_TIME","DEP_TIME_BLK","DEST","DEST_AIRPORT_ID","DEST_AIRPORT_SEQ_ID",
                 "DEST_CITY_MARKET_ID","DEST_CITY_NAME","DEST_STATE_ABR","DEST_STATE_FIPS","DEST_STATE_NM","DEST_WAC",
                 "DISTANCE","DISTANCE_GROUP","DIVERTED","FIRST_DEP_TIME","FL_DATE","LATE_AIRCRAFT_DELAY","LONGEST_ADD_GTIME",
                 "MONTH","NAS_DELAY","OP_CARRIER","OP_CARRIER_AIRLINE_ID","OP_CARRIER_FL_NUM","OP_UNIQUE_CARRIER",
                 "ORIGIN","ORIGIN_AIRPORT_ID","ORIGIN_AIRPORT_SEQ_ID","ORIGIN_CITY_MARKET_ID","ORIGIN_CITY_NAME",
                 "ORIGIN_STATE_ABR","ORIGIN_STATE_FIPS","ORIGIN_STATE_NM","ORIGIN_WAC","QUARTER","SECURITY_DELAY",
                 "TAIL_NUM","TAXI_IN","TAXI_OUT","TOTAL_ADD_GTIME","WEATHER_DELAY","WHEELS_OFF","WHEELS_ON","YEAR"]

# COMMAND ----------

df_temp = df_airlines.select(*selected_cols)

# COMMAND ----------

df_temp = df_temp.withColumn("DEST_CITY_MARKET_ID",col("DEST_CITY_MARKET_ID").cast(StringType())) \
        .withColumn("OP_CARRIER_AIRLINE_ID",col("OP_CARRIER_AIRLINE_ID").cast(StringType())) \
        .withColumn("OP_CARRIER_FL_NUM",col("OP_CARRIER_FL_NUM").cast(StringType())) \
        .withColumn("ORIGIN_AIRPORT_SEQ_ID",col("ORIGIN_AIRPORT_SEQ_ID").cast(StringType())) \
        .withColumn("DEST_AIRPORT_ID",col("DEST_AIRPORT_ID").cast(StringType())) \
        .withColumn("ARR_DELAY_GROUP",col("ARR_DELAY_GROUP").cast(StringType())) \
        .withColumn("DEP_DELAY_GROUP",col("DEP_DELAY_GROUP").cast(StringType())) \
        .withColumn("MONTH",col("MONTH").cast(StringType())) \
        .withColumn("DISTANCE_GROUP",col("DISTANCE_GROUP").cast(StringType())) \
        .withColumn("DEST_AIRPORT_SEQ_ID",col("DEST_AIRPORT_SEQ_ID").cast(StringType())) \
        .withColumn("ORIGIN_CITY_MARKET_ID",col("ORIGIN_CITY_MARKET_ID").cast(StringType())) \
        .withColumn("QUARTER",col("QUARTER").cast(StringType())) \
        .withColumn("YEAR",col("YEAR").cast(StringType())) \
        .withColumn("CANCELLED",col("CANCELLED").cast(BooleanType())) \
        .withColumn("DIVERTED",col("DIVERTED").cast(BooleanType()))

# COMMAND ----------

delay_cols = ["NAS_DELAY","LATE_AIRCRAFT_DELAY","SECURITY_DELAY","WEATHER_DELAY","CARRIER_DELAY"]

# fill empty values with 0 for delay columns. 
df_temp = df_temp.fillna(0, subset=delay_cols)

# COMMAND ----------

# removing null rows 
df_temp = df_temp.na.drop(subset=["DEP_DELAY"])

# COMMAND ----------

# Adjusting extreme imbalance in columns with null values and too many zeros (most of these are missing 96-99% rows)
missing_values_columns = ['LONGEST_ADD_GTIME', 'FIRST_DEP_TIME','TOTAL_ADD_GTIME','CANCELLATION_CODE','CANCELLED','DIVERTED']
zeros_columns = ['NAS_DELAY','SECURITY_DELAY','WEATHER_DELAY', 'LATE_AIRCRAFT_DELAY','DIV_AIRPORT_LANDINGS']
duplicated_cols = ['OP_CARRIER_AIRLINE_ID','DEST_STATE_ABR','ORIGIN_STATE_ABR']
other = ['YEAR']
missing_and_zeros_dups_other_cols = missing_values_columns + zeros_columns + duplicated_cols + other

df_temp = df_temp.drop(*missing_and_zeros_dups_other_cols)

# COMMAND ----------

print(len(df_temp.columns))
df_temp.printSchema()
# We have a total of 49 features + 1 target DEP_DEL15

# COMMAND ----------

# need to remove missing values to plot correlation
df_temp = df_temp.na.drop()
dbutils.data.summarize(df_temp)

# COMMAND ----------

# Separate target and independent variables
train = df_temp.filter(df_temp.MONTH < 3)
test = df_temp.filter(df_temp.MONTH >= 3)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# COMMAND ----------

# split the data before transformation
y_label = ['DEP_DEL15']
y_train, y_test = train.select(*y_label), test.select(*y_label) 

# define features
ohe_cols = ['ORIGIN','DAY_OF_WEEK','MONTH']
exclude = ohe_cols + ['DISTANCE_GROUP', 'MONTH']

categorical_cols = [t[0] for t in df_temp.dtypes if t[1] =='string']

non_catecorical_cols = [t[0] for t in df_temp.dtypes if t[1] !='string']
non_catecorical_cols.remove(*y_label)

x_features = categorical_cols + non_catecorical_cols
# x_train, x_test = train.select(*x_features), test.select(*x_features)
x_train, x_test = train, test

# confirm 49 features in the space
print(len(x_features))
x_train.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature colinearity or noise
# MAGIC Review what features are not correlated with our target variable or are too similar to other variables.

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import seaborn as sns
import  numpy as np

outcome = 'corr_features'

assembler = VectorAssembler(inputCols=non_catecorical_cols, outputCol=outcome)

air_vector = assembler.transform(x_train).select(outcome)

matrix = Correlation.corr(air_vector, outcome,'pearson').collect()[0][0]

corrmatrix = np.round(matrix.toArray(),3).tolist()
# print(corrmatrix)

# COMMAND ----------

# corrmatrix

# COMMAND ----------

# plot the heat map
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrmatrix, annot=True, fmt="g", cmap="YlGnBu", ax=ax)
plt.show()

# COMMAND ----------

# features to remove
correlated_ix = [1,3,4,8, 9, 13,14, 15,19,20,23]
[print(feature) for ix, feature in enumerate(non_catecorical_cols) if ix in correlated_ix]
new_non_catecorical_cols = [feature for ix, feature in enumerate(non_catecorical_cols) if ix not in correlated_ix]

print(len(non_catecorical_cols),
len(new_non_catecorical_cols))

# COMMAND ----------

# Update correlation matrix and review
outcome = 'corr_features'

assembler = VectorAssembler(inputCols=new_non_catecorical_cols, outputCol=outcome)

air_vector = assembler.transform(x_train).select(outcome)

matrix = Correlation.corr(air_vector, outcome,'pearson').collect()[0][0]

corrmatrix = np.round(matrix.toArray(),3).tolist()

# plot the heat map
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrmatrix, annot=True, fmt="g", cmap="YlGnBu", ax=ax)
plt.show()

# COMMAND ----------

# Drop correlated features
x_train = x_train.drop(*[feature for ix, feature in enumerate(non_catecorical_cols) if ix in correlated_ix])
x_test = x_test.drop(*[feature for ix, feature in enumerate(non_catecorical_cols) if ix in correlated_ix])

print(len(x_train.columns))
print(len(x_test.columns))

# COMMAND ----------



categorical_cols.remove('ORIGIN_CITY_NAME')
categorical_cols.remove('DEST_CITY_NAME')



# COMMAND ----------

# Let's take a look into the categorical variables first
display(x_train.select(*categorical_cols))

# COMMAND ----------

# we need to transform them into categorical values using frequency encoding
# The least frequent will get the lowest value = 0
# The most frequent will get the hights value = # categories
indexers=[]
new_categorical_cols=[]
for feature in categorical_cols:
    indexers.append(StringIndexer(inputCol=feature, outputCol=f"{feature}_indexed", stringOrderType='frequencyAsc'))
    new_categorical_cols.append(f"{feature}_indexed")
    
#Fits a model to the input dataset with optional parameters.
pipeline = Pipeline(stages=indexers)
x_train = pipeline.fit(x_train).transform(x_train).drop(*categorical_cols)


# COMMAND ----------

print(x_train.select(
'ARR_TIME_BLK_indexed').distinct().count())

# COMMAND ----------

# same for the test set
indexers_test=[]
# new_categorical_cols=[]
for feature in categorical_cols:
    indexers_test.append(StringIndexer(inputCol=feature, outputCol=f"{feature}_indexed", stringOrderType='frequencyAsc'))
#     new_categorical_cols.append(f"{feature}_indexed")
    
#Fits a model to the input dataset with optional parameters.
pipeline = Pipeline(stages=indexers)
x_test = pipeline.fit(x_test).transform(x_test).drop(*categorical_cols)
display(x_test.select(*new_categorical_cols))

# COMMAND ----------

# # 'DEP_TIME_BLK_indexed' - minBin = 32 and all indices are the different
# new_categorical_cols.remove('ARR_DELAY_GROUP_indexed')
# x_train = x_train.drop('ARR_DELAY_GROUP_indexed') 


for feature in new_categorical_cols:
    print(f"feature: {feature} - {x_train.select(feature).distinct().count()} of {x_train.select(feature).count()}")

# COMMAND ----------

# variables with excessive number of indexes
errors_rf = ['OP_CARRIER_FL_NUM_indexed', 'TAIL_NUM_indexed','DEST_indexed','DEST_AIRPORT_ID_indexed','DEST_AIRPORT_SEQ_ID_indexed','DEST_CITY_MARKET_ID_indexed',
            'DEST_STATE_NM_indexed','FL_DATE_indexed'] + ['ARR_DELAY_GROUP_indexed', 'ARR_TIME_BLK_indexed', 'DEP_DELAY_GROUP_indexed','DEP_TIME_BLK_indexed']

# Removed due to leakage
# ['ARR_DELAY_GROUP_indexed', 'ARR_TIME_BLK_indexed', DEP_DELAY_GROUP_indexed','DEP_TIME_BLK_indexed']

x_train = x_train.drop(*errors_rf)

# COMMAND ----------

x_test = x_test.drop('DEP_TIME_BLK_indexed')

# COMMAND ----------

new_categorical_cols = [cat for cat in new_categorical_cols if cat not in errors_rf]

# COMMAND ----------

# Cross Evaluator requires target variable to be renamed as label
x_train = x_train.withColumnRenamed('DEP_DEL15', 'label')
x_test = x_test.withColumnRenamed('DEP_DEL15', 'label')
display(x_train)

# COMMAND ----------


from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np


outcome='indexed_categories'
assembler = VectorAssembler(inputCols=new_categorical_cols, outputCol=outcome)

# Let's use random forest to find feature importance of categorical features
rf = RandomForestRegressor(featuresCol=outcome,
                           maxDepth=2,
                           numTrees=2,
                           maxBins=38,
                           labelCol='label',
#                            n_jobs=-1,
#                            oob_score=True,
                          bootstrap=True)

pipeline = Pipeline(stages=[assembler, rf])

paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 3, stop = 20, num = 3)]) \
    .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 3)]) \
    .build()



crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3)


cvModel = crossval.fit(x_train)
predictions = cvModel.transform(x_test)



# COMMAND ----------

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
rmse
# rfPred = cvModel.transform(df)
# rfResult = rfPred.toPandas()

# COMMAND ----------

display(predictions)

# COMMAND ----------

bestPipeline = cvModel.bestModel
bestModel = bestPipeline.stages[1]
importances = bestModel.featureImportances

sorted_indices = np.argsort(importances)[::-1]
sorted_importances = [importances[int(ix)] for ix in sorted_indices]
sorted_features = [new_categorical_cols[int(ix)] for ix in sorted_indices]

x_values = list(range(len(importances)))


fig, ax = plt.subplots(figsize=(20,10))

plt.bar(x_values, sorted_importances, orientation = 'vertical')
plt.xticks(x_values, sorted_features, rotation=40)
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.title('Feature Importances')

# COMMAND ----------

# indexed columns with too many indices
too_many_indices_cols = ['OP_CARRIER_FL_NUM_indexed', 'TAIL_NUM_indexed','DEST_indexed','DEST_AIRPORT_ID_indexed','DEST_AIRPORT_SEQ_ID_indexed','DEST_CITY_MARKET_ID_indexed',
            'DEST_STATE_NM_indexed','FL_DATE_indexed'] 

# COMMAND ----------

# function target encoding for features with too many indices
# Option 1
def target_mean_encoding(df, col, target):
    """
    :param df: pyspark.sql.dataframe
        dataframe to apply target mean encoding
    :param col: str list
        list of columns to apply target encoding
    :param target: str
        target column
    :return:
        dataframe with target encoded columns
    """
    target_encoded_columns_list = []
    for c in col:
        means = df.groupby(F.col(c)).agg(F.mean(target).alias(f"{c}_mean_encoding"))
        dict_ = means.toPandas().to_dict()
        target_encoded_columns = [F.when(F.col(c) == v, encoder)
                                  for v, encoder in zip(dict_[c].values(),
                                                        dict_[f"{c}_mean_encoding"].values())]
        target_encoded_columns_list.append(F.coalesce(*target_encoded_columns).alias(f"{c}_mean_encoding"))
    return df.select(target, *target_encoded_columns_list)


# function apply on spark inputs
df_target_encoded = target_mean_encoding(df, col=['cate1'], target='label')

df_target_encoded.show()

# COMMAND ----------



# COMMAND ----------

# #option 2
# import h2o
# from pysparkling.ml import H2OTargetEncoder
# from pysparkling.ml import H2OGBMClassifier
# from pyspark.ml import Pipeline

# targetEncoder = H2OTargetEncoder()\
#   .setInputCols(too_many_indices_cols)\
#   .setLabelCol("CAPSULE")\
#   .setProblemType("Classification")

# gbm = H2OGBMClassifier()\
#     .setFeaturesCols(targetEncoder.getOutputCols())\
#     .setLabelCol("CAPSULE")
# pipeline = Pipeline(stages=[targetEncoder, gbm])

# pipelineModel = pipeline.fit(x_train)
# pipelineModel.transform(x_test).show()

# from pyspark.ml import PipelineModel
# pipelineModel.save("somePathForStoringPipelineModel")
# loadedPipelineModel = PipelineModel.load("somePathForStoringPipelineModel")
# loadedPipelineModel.transform(testingDF).show()

# COMMAND ----------

len(x_train.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature transformation
# MAGIC We are dividing the data between categorical features and numerical.
# MAGIC 
# MAGIC Categorical features will be transformed into frequency encoding and one hot encoding.
# MAGIC 
# MAGIC Numerical features and Frequency features will be standardized to center their distribution in zero. Binary variables will remain as boolean.

# COMMAND ----------

# # Get all the categorical variables
# categorical_cols = [t[0] for t in df_temp.dtypes if t[1] =='string']

# # Separate column names based on encoding type
# # We would need to do training and tuning to test this classifier and install it
# # This will be skipped for the baseline and can be explored further later
# lgbm_cols = ['MONTH','DEP_TIME_BLK', 'ARR_TIME_BLK']

# # I removed QUARTER because it only has one value
# # information in Origin and Origin abreviation is the same, one of both could be dropped
# ohe_cols = ['ORIGIN']

# Features that require Frequency encoding
fe_cols = [feature for feature in categorical_cols if feature not in exclude]
fe_cols

# COMMAND ----------

# MAGIC %md
# MAGIC # Frequency encoding
# MAGIC All string variables are renamed to the {variable}_COUNT.
# MAGIC 
# MAGIC First, we create a table with counts and frequencies. Then we join in the new colum the frequency corresponding the key. Finally, we drop the original feature.

# COMMAND ----------

# df_with_fe_train = x_train
# df_with_fe_train = x_train.select(*fe_cols)
# print("Number of feature before conversion: ",len(df_with_fe_train.columns))
# df_with_fe_test = x_test
# df_with_fe_test = x_test.select(*fe_cols)


# def update_df_with_fe(df_to_encode, col_to_encode):
    
#     # create a temporary frequency table for each feature
#     print(col_to_encode)
#     feature_count_label = f'{col_to_encode}_COUNT'
#     feature_frequency_train = df_to_encode.groupBy(col_to_encode).count().withColumnRenamed("count", feature_count_label)

#     # Join in temp frequency table to main dataframe
#     df_with_fe = df_to_encode.join(feature_frequency_train, feature).drop(feature)
#     return df_with_fe

# # train set
# for feature in fe_cols:
#     x_train = update_df_with_fe(x_train, feature)
# # test set
# for feature in fe_cols:
#     x_test = update_df_with_fe(x_test, feature)

# for feature in fe_cols:
#     print(feature)
#     feature_count_label = f'{feature}_COUNT'
# #     feature_frequency_train = df_with_fe_train.groupBy(feature).count().withColumnRenamed("count", feature_count_label)
#     feature_frequency_test = df_with_fe_test.groupBy(feature).count().withColumnRenamed("count", feature_count_label)
    
    
#     # Join in temp frequency table to main dataframe
# #     df_with_fe_train = df_with_fe_train.join(feature_frequency_train, feature).drop(feature)
#     df_with_fe_test = df_with_fe_test.join(feature_frequency_train, feature).drop(feature)
# # print("Number of feature after conversion: ",len(fe_train.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## One Hot Encoder
# MAGIC In this case, the Origin data include only two airports, so the current encoder includes only two options 0,1. When we have all the data this would increase to the number of airports of origin.
# MAGIC 
# MAGIC Output: variables are in a single column as a vector (needs to be expanded) with the corresponding column names.

# COMMAND ----------

# Steps to one hot encode variables
# Indexers converts target variable into indices (e.g. for 3 variables -> 0,1,2)
indexer = [StringIndexer(inputCol= feature_name,
                       outputCol= f"{feature_name}_INDEX")
          for feature_name in ohe_cols]

# Encoder takes the indices and encodes as 0 or 1 if variable present
encoders = [OneHotEncoder(dropLast=False,                
                        inputCol= f"{feature_name}_INDEX",
                        outputCol=f"{feature_name}_ENCODED")
           for feature_name in ohe_cols]
# creates a sparse vector with all the columns, in this case contains 4 values (2 columns in 2 variables)
assembler = VectorAssembler(
  inputCols=[f"{encoder.getOutputCol()}" 
             for encoder in encoders],
  outputCol="oheFeatures"
)

# the pipeline processes all the changes and outputs the column of the assembler "features" with the final vector. This column is the one that would require to be expanded
pipeline = Pipeline(stages=indexer + encoders + [assembler])
model = pipeline.fit(df_with_fe_features)
df_with_fe_ohe_vectors = model.transform(df_with_fe_features)
df_with_fe_ohe_vectors = df_with_fe_ohe_vectors.drop(*ohe_cols)                       

# COMMAND ----------

# MAGIC %md
# MAGIC Converting vector output of OHE into columns

# COMMAND ----------

# Get one row of the DF to inspect the vector size (potentially a more efficient way to do this by pulling it out of the model above, but I couldn't work out how)
row = df_with_fe_ohe_vectors.select("oheFeatures").head(1)[0]
# Get the size of the vector so we know how many columns to extract
vec_size = row["oheFeatures"].size
# Convert to array
df_with_fe_ohe_vectors = df_with_fe_ohe_vectors.withColumn("oheFeatures_SPLT", vector_to_array("oheFeatures"))
# Convert array to cols
for i in range(vec_size):
    df_with_fe_ohe_vectors = df_with_fe_ohe_vectors.withColumn(f"oheFeatures_{i}", col("oheFeatures_SPLT")[i])
df_with_fe_ohe_vectors = df_with_fe_ohe_vectors.drop("oheFeatures_SPLT")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rescaling of variables - log transformations
# MAGIC Based on the initial EDA we detected some features that required rescaling due to extreme skewness.

# COMMAND ----------

# Log transformation
log_features = ['DEP_DELAY_NEW', 'TAXI_OUT', 'TAXI_IN', 'ARR_DELAY_NEW',
'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE']

# Frequency encoding + one hot encoding labels that require transformation
fe_ohe_labels  = df_with_fe_ohe_vectors.columns

log_features = [log_f for log_f in log_features if log_f in fe_ohe_labels]

# COMMAND ----------

#Transformed the columns. To avoid nulls we added a small value above 0.
for feature in log_features:
  df_with_fe_ohe_vectors = df_with_fe_ohe_vectors.withColumn(f"log_{feature}", F.log(F.col(feature)+0.000001)).drop(feature)

# DEP_DELAYS and ARR_DELAYS include negative numbers, so these were excluded

# COMMAND ----------

# Need to remove these string variables after being ohe
drop_features = ['ORIGIN', 'ORIGIN_STATE_ABR']

df_with_fe_ohe_vectors = df_with_fe_ohe_vectors.drop(*drop_features)

# COMMAND ----------

# Keep non scalable/scalable features separated for standardization
non_scalable_features = ['ARR_DEL15', 'DEP_DEL15','ORIGIN_INDEX', 'ORIGIN_STATE_ABR_INDEX', 'ORIGIN_ENCODED', 'ORIGIN_STATE_ABR_ENCODED',
                        'oheFeatures_0',  'oheFeatures_1',  'oheFeatures_2',  'oheFeatures_3'] 

all_features = df_with_fe_ohe_vectors.columns
scalable_features = [feature for feature in all_features if feature not in non_scalable_features]
# scalable_features

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Standardizing (val-mean)/std 
# MAGIC This process will create a column with a vector "features" with all the standardized values. This column will require expansion into multiple columns.

# COMMAND ----------

# Scaler function
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
# vector creation - it could be possible to create a single pipeline instead of two, but I haven't figured it out
assembler = VectorAssembler().setInputCols(scalable_features).setOutputCol("features")

features_transformed = assembler.transform(df_with_fe_ohe_vectors)

scaler_model = scaler.fit(features_transformed.select("features"))
features_scaled = scaler_model.transform(features_transformed)

# COMMAND ----------

df_with_fe_ohe_vectors.columns

# COMMAND ----------

#output frame
display(features_scaled)

# COMMAND ----------

y_label = ['DEP_DEL15']
x_labels = ['ARR_DEL15',
 'ARR_DELAY',
 'ARR_TIME',
 'CARRIER_DELAY',
 'CRS_ARR_TIME',
 'CRS_DEP_TIME',
 'DAY_OF_MONTH',
 'DAY_OF_WEEK',
 'DEP_DEL15',
 'DEP_DELAY',
 'DEP_TIME',
 'DEST_STATE_FIPS',
 'DEST_WAC',
 'ORIGIN_AIRPORT_ID',
 'ORIGIN_STATE_FIPS',
 'ORIGIN_WAC',
 'WHEELS_OFF',
 'WHEELS_ON',]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC 1. Expand the vectors oheFeatures and features into individual columns.
# MAGIC > oheFeatures = column names -> ohe_features
# MAGIC >
# MAGIC > features = column names -> scalable_features
# MAGIC 2. Join these features with "non_scalable_features" to create a single frame.
# MAGIC 
# MAGIC 3. Write the transformed dataset in our blob

# COMMAND ----------

# # SAS Access to blob container
# spark.conf.set(
#   f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
#   dbutils.secrets.get(scope = secret_scope, key = secret_key)
# )

# #write the parquet file
# features_scaled.write.parquet(f"{blob_url}/transformed_dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Missing values
# MAGIC Before imputing features we must verify that the target value exists in such feature.
# MAGIC 
# MAGIC If the target value is missing, then we should drop those rows.

# COMMAND ----------

missing_data = ['ACTUAL_ELAPSED_TIME', 'ARR_DELAY','TAXI_IN', 'ARR_TIME','AIR_TIME','ARR_DEL15','ARR_DELAY_NEW','TAXI_OUT','WHEELS_OFF','WHEELS_ON','ARR_DELAY_GROUP']

for feature in missing_data:

  null_feature = df_temp.filter(df_temp[feature].isNull()).select(feature).count()

  null_target = df_temp.filter(df_temp[feature].isNull()).select(df_temp.ARR_DEL15.isNull()).count()

  print(f"{feature} has {100*null_target/null_feature}% missing target variable values")

# COMMAND ----------

# MAGIC %md
# MAGIC Conclusion: the missing values in the df_dataset are less the than 0.35% of the data. After comparing with the target value DEP_DEL15, we found that the null values are present in both feature and target variable. 

# COMMAND ----------

# MAGIC %md
# MAGIC # 6 month feature selection

# COMMAND ----------

# MAGIC %run "./libs/time_based_features"

# COMMAND ----------

# Inspect the Mount's Final Project folder 
# display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

df_joined = spark.read.parquet(f"{blob_url}/join_6m_0329")


# COMMAND ----------

# display(df_joined)

# COMMAND ----------

# features that carry leakage - exclude
leakage = ['ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'ARR_DEL15', 'ARR_DELAY', 'ARR_DELAY_GROUP', 'ARR_DELAY_NEW','ARR_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DEP_DELAY', 'DEP_DELAY_GROUP', 'DEP_DELAY_NEW', 'DEP_TIME', 'DEP_TIME_BLK','CRS_DEP_TIME']

# features related to datetime needed to exclude from regression
date_time = ['DAY_OF_WEEK','FL_DATE','YEAR'
             , 'TIMESTAMP_UTC','WEATHER_WINDOW_START','WEATHER_WINDOW_END' ]

# features used to extract weather data - exclude
stations = ['coordinates', 'station_id','lat','lon','neighbor_id','neighbor_name','neighbor_state','neighbor_call',
           'distance_to_neighbor','dist_to_airport_rank','STATION','DATE', 'SOURCE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME','CALL_SIGN','DATE_UTC']

# duplicate feature information - exclude
duplicate = ['DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','DEST_CITY_NAME','DEST_STATE_FIPS','DEST_STATE_NM','DEST_WAC', 'OP_CARRIER_AIRLINE_ID','OP_UNIQUE_CARRIER','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','ORIGIN_STATE_FIPS' , 'ORIGIN_STATE_NM', 'ORIGIN_WAC','ident','ORIGIN_CITY_NAME']

# cause of delay is also information leakage - exclude
cause_delay = ['NAS_DELAY','OP_CARRIER','SECURITY_DELAY','CARRIER_DELAY']

# can be used to get new features - exclude
identifiers = ['OP_CARRIER_FL_NUM','TAIL_NUM','iata_code','TIMESTAMP', 'DAY_OF_MONTH','MONTH']

# features related to origin and destination
location = ['ORIGIN','DEST', 'ORIGIN_CITY_MARKET_ID','DEST_CITY_MARKET_ID', 'DEST_STATE_ABR']

featuresToExclude = leakage + date_time + stations + duplicate + cause_delay 

# COMMAND ----------

# drop unecessary features
df_joined = df_joined.drop(*featuresToExclude)

# COMMAND ----------

# dbutils.data.summarize(df_joined)

# COMMAND ----------



'''
all values are the same and have no missing values
'''
constant_value_features = ['WND_SpeedQuality','VIS_DistanceQuality','DEW_Quality','TMP_Quality','WND_DirectionQuality','CARRIER_DELAY','VIS_VariabilityQuality', 'REPORT_TYPE']

'''
missing values contain over 99% missing data
*CIG_CeilingAndVisibilityOK it's mainly 'N' meaning status wasn't reported
*CIG_CeilingDetermination method used to create the report, most is M: Measured
*AA1_RainCondition methos used to measue, most is missing, second is 'trace'
'''
missing_values = ['WND_SpeedQuality','AL1_SnowAccumQuality','AA4_RainQuality'  ,'AL2_SnowAccumQuality','AL3_SnowAccumDuration','AL3_SnowAccumDepth'
                  ,'AW4_PresentWeatherQuality','AL2_SnowAccumCondition','AA4_RainCondition'
                  ,'AJ1_SnowEqWaterDepthCondition','AJ1_SnowDepthCondition','CIG_CeilingAndVisibilityOK'
                  ,'AL3_SnowAccumCondition','AL_SnowAccumDuration','CIG_CeilingDetermination'
                  ,'AA1_RainCondition','AL_SnowAccumDepth','AA3_RainCondition'
                  ,'AA2_RainCondition','AL1_SnowAccumCondition'] #+ ['ORIGIN_CITY_NAME','CRS_DEP_TIME']

'''
contain the code of weather type
'''
code_expansion = ['AW4_PresentWeatherCond','AW2_PresentWeatherCond','AW3_PresentWeatherCond','AW1_PresentWeatherCond']

'''
string to number
'''
toInt=['AA_RainDepth']

'''
median, mean for that day at origin location
'''
fillWithMedian =['CIG_CeilingHeightDim','VIS_Horizontal','WND_DirectionAngle','DEW_Value']

fillWithMean = ['WND_Speed','TMP_Value','SLP_Value']

'''
AW2_PresentWeatherQuality (AW1,AW3) - 1 passed quality status, 0 not recorded
CIG_CeilingHeightDim - distance to clouds, if no clouds then zero
AL3_SnowAccumQuality - amount of snow, if no snow, then zero
CIG_CeilingQuality - 1 passed quality status, if zero then no clouds or didn't pass
SLP_Quality - 1 passed quality status, if zero then didn't pass or wasn't recorded
AJ1_SnowEqWaterDepth - if no snow, then depth is zero
AJ1_SnowEqWaterDepthQuality - 1 passed quality status, zero didn't pass or wasn't recorded
AA2_RainQuality - 1 passed status, zero didn't pass or wasn't recorded
AA1_RainQuality - same as above
AJ1_SnowDepthQuality - 1 passed quality status, if zero then no snow or didn't pass
AA_RainDepth - 0 no rain
''' 
fillWithZero = ['AW2_PresentWeatherQuality', 'AW1_PresentWeatherQuality', 'AW3_PresentWeatherQuality'
                ,'CIG_CeilingHeightDim','AL3_SnowAccumQuality','CIG_CeilingQuality'
                ,'SLP_Quality', 'AJ1_SnowEqWaterDepth','AJ1_SnowEqWaterDepthQuality'
                ,'AA2_RainQuality' ,'AA1_RainQuality','AJ1_SnowDepthQuality'
                ,'AA_RainDuration','VIS_Variability','AA_RainDepth','AA3_RainQuality']



# COMMAND ----------

# Drop features with extreme missing values 
featuresToExclude = constant_value_features + missing_values
df_joined = df_joined.drop(*featuresToExclude)

# COMMAND ----------

# Fill with zero
df_joined = df_joined.fillna(0, subset=fillWithZero)

# COMMAND ----------

# df_join_agg = df_joined.groupBy("OP_CARRIER_FL_NUM", "TAIL_NUM", "TIMESTAMP") \
#                            .agg(*(max("DEP_DEL15").alias("DEP_DEL15"),  \

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, BooleanType, IntegerType

# COMMAND ----------

w = Window.partitionBy([F.col(x) for x in identifiers])

for feature in fillWithMean:
    df_joined = df_joined.withColumn(f"{feature}_mean"
                                     , when(col(feature).isNull(),
                                            F.avg(df_joined[feature]).over(w))
                                     .otherwise(df_joined[feature])).drop(feature)


# COMMAND ----------

df_joined = df_joined.withColumn("AA_RainDepth",col("AA_RainDepth").cast(IntegerType()))

# COMMAND ----------

for feature in fillWithMedian:
    df_joined = df_joined.withColumn(f"{feature}_median"
                                     , when(col(feature).isNull(),
                                            F.percentile_approx(df_joined[feature],0.5).over(w)) \
                                     .otherwise(df_joined[feature])).drop(feature)
    
# df_joined=df_joined.drop("CIG_CeilingHeightDim_mean")

# COMMAND ----------

# Let's join all the conditions in a single column
df_joined = df_joined.withColumn("weatherCondition", 
                    when(col('AW1_PresentWeatherCond').isNull(), df_joined['AW2_PresentWeatherCond']) \
#                      .when(df_joined['AW2_PresentWeatherCond'].isNull(),df_joined['AW3_PresentWeatherCond']) \
#                      .when(df_joined['AW3_PresentWeatherCond'].isNull(),df_joined['AW4_PresentWeatherCond']) \
                     .otherwise(df_joined['AW1_PresentWeatherCond'])) \

#          .drop(code_expansion) \
#          .fillna(0, subset=["weatherCondition"]) \


# COMMAND ----------

df_joined = df_joined.drop(*code_expansion)

# COMMAND ----------

df_joined = df_joined.fillna(0,"weatherCondition")
df_joined = df_joined.fillna(0,"WND_DirectionAngle_median")
df_joined = df_joined.fillna(0,"AA_RainDepth")
# df_joined = df_joined.fillna(0,"AJ1_SnowDepth")

# COMMAND ----------

train.columns

# COMMAND ----------

df_joined = df_joined.withColumn("weatherCondition",col("weatherCondition").cast(StringType()))


# COMMAND ----------

# Get numerical and categorical values
y_label = ['DEP_DEL15']
categorical_cols = [t[0] for t in df_joined.dtypes if t[1] =='string']

non_catecorical_cols = [t[0] for t in df_joined.dtypes if t[1] !='string']
non_catecorical_cols.remove(*y_label)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split into train and test

# COMMAND ----------

train = df_joined.filter((df_joined.MONTH < 5) | ((df_joined.MONTH==5) & (df_joined.DAY_OF_MONTH <= 15)))
test = df_joined.filter((df_joined.MONTH == 6) | ((df_joined.MONTH==5) & (df_joined.DAY_OF_MONTH > 15)))

# COMMAND ----------

train = train.drop(*identifiers)
test = test.drop(*identifiers)

# COMMAND ----------

non_catecorical_cols.remove('DAY_OF_MONTH')
non_catecorical_cols.remove('TIMESTAMP')

# COMMAND ----------

# Check for missing values before running correlation
# dbutils.data.summarize(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation matrix

# COMMAND ----------

# Log transformation
log_features = ['CIG_CeilingHeightDim_median','AJ1_SnowDepth','WND_Speed_mean','DISTANCE']

#Transformed the columns. To avoid nulls we added a small value above 0.
for feature in log_features:
    df_joined = df_joined.withColumn(f"{feature}", F.log(F.col(feature)+0.000001)).drop(feature)

# COMMAND ----------

correlatedFeatures = ['SLP_Quality','AL3_SnowAccumQuality','AJ1_SnowEqWaterDepthQuality','AJ1_SnowDepth'] + code_expansion
non_catecorical_cols = [feature for feature in non_catecorical_cols if feature not in correlatedFeatures]
train = train.drop(*correlatedFeatures)
test = test.drop(*correlatedFeatures)

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import seaborn as sns
import  numpy as np

outcome = 'corr_features'

assembler = VectorAssembler(inputCols=non_catecorical_cols+['label'], outputCol=outcome)

air_vector = assembler.transform(train).select(outcome)

matrix = Correlation.corr(air_vector, outcome,'pearson').collect()[0][0]

corrmatrix = np.round(matrix.toArray(),3).tolist()

# COMMAND ----------

# plot the heat map
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrmatrix, annot=True, fmt="g", cmap="YlGnBu", ax=ax, xticklabels=non_catecorical_cols+['label'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance of categorical variables

# COMMAND ----------

# review missing data
# dbutils.data.summarize(train)

train = train.fillna("0", "AA_RainDuration")
test = test.fillna("0", "AA_RainDuration")

train = train.fillna("N", "VIS_Variability")
test = test.fillna("N", "VIS_Variability")


# COMMAND ----------

# remove identifiers from categorical variables
# categorical_cols.remove("MONTH")
# categorical_cols.remove("OP_CARRIER_FL_NUM")
# categorical_cols.remove("TAIL_NUM")
categorical_cols.remove("iata_code")

# COMMAND ----------

# we need to transform them into categorical values using frequency encoding
# The least frequent will get the lowest value = 0
# The most frequent will get the hights value = # categories
indexers=[]
new_categorical_cols=[]
for feature in categorical_cols:
    indexers.append(StringIndexer(inputCol=feature, outputCol=f"{feature}_indexed", stringOrderType='frequencyAsc'))
    new_categorical_cols.append(f"{feature}_indexed")
    
#Fits a model to the input dataset with optional parameters.
pipeline = Pipeline(stages=indexers)
train = pipeline.fit(train).transform(train).drop(*categorical_cols)
test = pipeline.fit(test).transform(test).drop(*categorical_cols)

# COMMAND ----------

for feature in new_categorical_cols:
    print(f"feature: {feature} - {x_train.select(feature).distinct().count()} of {x_train.select(feature).count()}")

# COMMAND ----------

targetEncodig = ['DEST_indexed', 'DEST_CITY_MARKET_ID_indexed', 'DEST_STATE_ABR_indexed']
new_categorical_cols = [cat for cat in new_categorical_cols if cat not in targetEncodig]

# COMMAND ----------

# Cross Evaluator requires target variable to be renamed as label
# train = train.withColumnRenamed('DEP_DEL15', 'label')
# test = test.withColumnRenamed('DEP_DEL15', 'label')
display(train)

# COMMAND ----------

x_train = x_train.drop(*targetEncodig)

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np

x_train = train
x_test = test

outcome='indexed_categories'
assembler = VectorAssembler(inputCols=new_categorical_cols, outputCol=outcome)

# Let's use random forest to find feature importance of categorical features
rf = RandomForestRegressor(featuresCol=outcome,
                           maxDepth=2,
                           numTrees=2,
                           maxBins=38,
                           labelCol='label',
#                            n_jobs=-1,
#                            oob_score=True,
                          bootstrap=True)

pipeline = Pipeline(stages=[assembler, rf])

paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 3, stop = 50, num = 3)]) \
    .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 3)]) \
    .build()



crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=7)


cvModel = crossval.fit(x_train)
predictions = cvModel.transform(x_test)

# COMMAND ----------

bestPipeline = cvModel.bestModel
bestModel = bestPipeline.stages[1]
importances = bestModel.featureImportances

sorted_indices = np.argsort(importances)[::-1]
sorted_importances = [importances[int(ix)] for ix in sorted_indices]
sorted_features = [new_categorical_cols[int(ix)] for ix in sorted_indices]

x_values = list(range(len(importances)))


fig, ax = plt.subplots(figsize=(20,10))

plt.bar(x_values, sorted_importances, orientation = 'vertical')
plt.xticks(x_values, sorted_features, rotation=40)
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.title('Feature Importances')

# COMMAND ----------

# used for normal regression
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
rmse


from pyspark.ml.evaluation import BinaryRegressionEvaluator
# used for binary classification
evaluator = BinaryRegressionEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
auc

# COMMAND ----------

sorted_features

# COMMAND ----------

