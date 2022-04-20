# Databricks notebook source
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

# COMMAND ----------

# PRIVATE: Load airlines data from file system
def _load_3month_airlines_data():
    '''
    Private function to fetch the 3 month airline data from storage
    Outputs: Spark DataFrame
    '''
    df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/").cache()
    return df_airlines

def _load_6month_airlines_data():
    '''
    Private function to fetch the 6 month airline data from storage
    Outputs: Spark DataFrame
    '''
    df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_6m/").cache()
    return df_airlines

def _load_full_airlines_data_train():
    '''
    Private function to fetch the 2015-2018 airline data for training from storage
    Outputs: Spark DataFrame
    '''
    df_airlines_2015 = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2015.parquet/*").cache()
    df_airlines_2016 = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2016.parquet/*").cache()
    df_airlines_2017 = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2017.parquet/*").cache()
    df_airlines_2018 = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2018.parquet/*").cache()
    df_airlines_train = df_airlines_2015.unionByName(df_airlines_2016, allowMissingColumns = True).unionByName(df_airlines_2017, allowMissingColumns = True).unionByName(df_airlines_2018, allowMissingColumns = True).cache()
    return df_airlines_train

def _load_full_airlines_data_test():
    '''
    Private function to fetch the 2019 airline data for testing from storage
    Outputs: Spark DataFrame
    '''
    df_airlines_2019 = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2019.parquet/*").cache()
    return df_airlines_2019

# COMMAND ----------

# PRIVATE: Get airlines columns constants

def _get_airlines_cols():
    '''
    Private function to get the airline columns we are interested in 
    Outputs: List
    '''
    selected_cols = ["ACTUAL_ELAPSED_TIME","AIR_TIME","ARR_DEL15","ARR_DELAY","ARR_DELAY_GROUP","ARR_DELAY_NEW","ARR_TIME",
                 "ARR_TIME_BLK","CARRIER_DELAY","CRS_ARR_TIME","CRS_DEP_TIME",
                 "CRS_ELAPSED_TIME","DAY_OF_MONTH","DAY_OF_WEEK","DEP_DEL15","DEP_DELAY","DEP_DELAY_GROUP",
                 "DEP_DELAY_NEW","DEP_TIME","DEP_TIME_BLK","DEST","DEST_AIRPORT_ID","DEST_AIRPORT_SEQ_ID",
                 "DEST_CITY_MARKET_ID","DEST_CITY_NAME","DEST_STATE_ABR","DEST_STATE_FIPS","DEST_STATE_NM","DEST_WAC",
                 "DISTANCE","DISTANCE_GROUP","FL_DATE",
                 "MONTH","NAS_DELAY","OP_CARRIER","OP_CARRIER_AIRLINE_ID","OP_CARRIER_FL_NUM","OP_UNIQUE_CARRIER",
                 "ORIGIN","ORIGIN_AIRPORT_ID","ORIGIN_AIRPORT_SEQ_ID","ORIGIN_CITY_MARKET_ID","ORIGIN_CITY_NAME",
                 "ORIGIN_STATE_ABR","ORIGIN_STATE_FIPS","ORIGIN_STATE_NM","ORIGIN_WAC","QUARTER","SECURITY_DELAY",
                 "TAIL_NUM","YEAR"]
    return selected_cols

# COMMAND ----------

# PRIVATE: Select airlines columns
def _cast_airlines_cols(df):
    '''
    Private function to cast the airlines columns to more appropriate data types
    Inputs: Spark DataFrame containing airlines data
    Outputs: Spark DataFrame
    '''
    df = df.withColumn("DEST_CITY_MARKET_ID",F.col("DEST_CITY_MARKET_ID").cast(StringType())) \
        .withColumn("OP_CARRIER_AIRLINE_ID",F.col("OP_CARRIER_AIRLINE_ID").cast(StringType())) \
        .withColumn("OP_CARRIER_FL_NUM",F.col("OP_CARRIER_FL_NUM").cast(StringType())) \
        .withColumn("ORIGIN_AIRPORT_SEQ_ID",F.col("ORIGIN_AIRPORT_SEQ_ID").cast(StringType())) \
        .withColumn("DEST_AIRPORT_ID",F.col("DEST_AIRPORT_ID").cast(StringType())) \
        .withColumn("ARR_DELAY_GROUP",F.col("ARR_DELAY_GROUP").cast(StringType())) \
        .withColumn("DEP_DELAY_GROUP",F.col("DEP_DELAY_GROUP").cast(StringType())) \
        .withColumn("MONTH",F.col("MONTH").cast(StringType())) \
        .withColumn("DISTANCE_GROUP",F.col("DISTANCE_GROUP").cast(StringType())) \
        .withColumn("DEST_AIRPORT_SEQ_ID",F.col("DEST_AIRPORT_SEQ_ID").cast(StringType())) \
        .withColumn("ORIGIN_CITY_MARKET_ID",F.col("ORIGIN_CITY_MARKET_ID").cast(StringType())) \
        .withColumn("QUARTER",F.col("QUARTER").cast(StringType())) \
        .withColumn("YEAR",F.col("YEAR").cast(StringType()))
    return df

def _parse_airlines_cols(df):
    '''
    Private function select and parse columns in the airlines data
    Inputs: Spark DataFrame containing airlines data
    Outputs: Spark DataFrame
    '''
    cols = _get_airlines_cols()
    df = df.select(*cols)
    df = _cast_airlines_cols(df)
    return df

# COMMAND ----------

# PRIVATE: Drop duplicate rows
def _drop_duplicates_airlines(df, verbose = False):
    '''
    Private function to drop duplicate rows in the airlines data
    Inputs: Spark DataFrame containing airlines data
    Outputs: Spark DataFrame
    '''
    if verbose: pre_count = df.count()
    df = df.dropDuplicates(["FL_DATE", "CRS_DEP_TIME", "DEST_AIRPORT_ID", "ORIGIN_AIRPORT_ID"])
    if verbose: 
        post_count = df.count()
        print(f"Dropped {pre_count - post_count} duplicate rows, which is {round(((pre_count - post_count) / post_count) * 100, 2)}% of the data")
    return df


# COMMAND ----------

# PRIVATE: Filling delay indicator

def _get_airlines_delay_cols():
    '''
    Private function to return the airline delay columns
    Outputs: List
    '''
    delay_cols = ["NAS_DELAY","SECURITY_DELAY","CARRIER_DELAY"]
    return delay_cols

def _fill_empty_airlines_delay_cols(df):
    '''
    Private function to fill the airlines delay columns with 0 if they are empty
    Inputs: Spark DataFrame containing airlines data
    Outputs: Spark DataFrame
    '''
    delay_cols = _get_airlines_delay_cols()
    df = df.fillna(0, subset=delay_cols)
    return df

# COMMAND ----------

# PRIVATE: Drop data where DEP_DELAY is Null

def _drop_na_dep_delay(df):
    '''
    Private function that drops rows where DEP_DELAY is null
    Inputs: Spark DataFrame containing airlines data
    Outputs: Spark DataFrame
    '''
    df = df.na.drop(subset=["DEP_DELAY"])
    return df

# COMMAND ----------

# PRIVATE: Putting it all together
def _parse_and_clean_airlines_data(df, verbose = False):
    '''
    Private function to parse the airlines data
    Inputs: Spark DataFrame containing airlines data
    Outputs: Spark DataFrame
    '''
    df = _parse_airlines_cols(df)
    df = _drop_duplicates_airlines(df, verbose = verbose)
    df = _fill_empty_airlines_delay_cols(df)
    df = _drop_na_dep_delay(df)
    return df

# COMMAND ----------

# PUBLIC: Load and clean airlines data

def get_3month_airlines_data(verbose = False):
    '''
    Get the parsed 3 months airlines data
    Outputs: Spark DataFrame
    '''
    df = _load_3month_airlines_data()
    df = _parse_and_clean_airlines_data(df, verbose = verbose)
    return df

def get_6month_airlines_data(verbose = False):
    '''
    Get the parsed 6 months airlines data
    Outputs: Spark DataFrame
    '''
    df = _load_6month_airlines_data()
    df = _parse_and_clean_airlines_data(df, verbose = verbose)
    return df
  
def get_full_airlines_data_train(verbose = False):
    '''
    Get the parsed training airlines data
    Outputs: Spark DataFrame
    '''
    df = _load_full_airlines_data_train()
    df = _parse_and_clean_airlines_data(df, verbose = verbose)
    return df

def get_full_airlines_data_test(verbose = False):
    '''
    Get the parsed test airlines data
    Outputs: Spark DataFrame
    '''
    df = _load_full_airlines_data_test()
    df = _parse_and_clean_airlines_data(df, verbose = verbose)
    return df