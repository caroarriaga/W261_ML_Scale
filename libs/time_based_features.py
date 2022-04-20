# Databricks notebook source
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import airporttime
from datetime import datetime, timedelta

# COMMAND ----------

def add_previous_flight_delay_indicator(df):
    '''
    Takes a data frame and appends a column for each aircraft flight indicating if the previous flight for that aircraft was delayed or not
    Inputs: Spark DataFrame (must contain TAIL_NUM, TIMESTAMP_UTC and DEP_DEL15 columns)
    Outputs: Spark DataFrame with column appended
    '''
    window = Window.partitionBy(['TAIL_NUM']).orderBy('TIMESTAMP_UTC')
        
    df = df.withColumn(
            'PREV_TIMESTAMP_UTC',
            F.lag(F.col('TIMESTAMP_UTC')).over(window)
        )
    
    def check_window(current_timestamp, previous_timestamp):
        if previous_timestamp is not None:
            # Previous flight's planned departure was more then 2 hours before next flight's departure, but less than 12 hours
            return ((current_timestamp - timedelta(hours=2)) >= previous_timestamp) and ((current_timestamp - timedelta(hours=12)) < previous_timestamp)
        else: 
            return False
 
    is_usable_time = udf(lambda timestamps: check_window(timestamps[0], timestamps[1]))
    
    df = df.withColumn(
            'USABLE',
            is_usable_time(F.array(F.col('TIMESTAMP_UTC'), F.col('PREV_TIMESTAMP_UTC')))
        )
    
    df = df.withColumn(
        'PREV_DEP_DEL15_TEMP',
        F.lag(F.col('DEP_DEL15')).over(window)
    ) \
        .withColumn(
        'PREV_DEP_DEL15', 
        F.when(F.col('USABLE') == True, F.col('PREV_DEP_DEL15_TEMP')).otherwise(F.lit(0))
    ) \
        .drop(*['PREV_DEP_DEL15_TEMP', 'USABLE', 'PREV_TIMESTAMP_UTC'])
    return df

# COMMAND ----------

### Still need to fix adding 2 hours before flight time

def add_airline_airport_status_indicator(df):
    '''
    Takes a data frame and appends a column for each aircraft flight indicating the % of delayed flights by airport and airline 
    Params: dataframe (must contain OP_UNIQUE_CARRIER, ORIGIN, TAIL_NUM, TIMESTAMP_UTC and DEP_DEL15 columns)
    Returns: dataframe with column appended
    '''
    window = Window.partitionBy(['OP_UNIQUE_CARRIER', 'ORIGIN']).orderBy('TIMESTAMP_UTC').rowsBetween(-20, -5)
          
    df = df.withColumn('AIRLINE_AIRPORT_STATUS', F.mean(F.col('DEP_DEL15')).over(window))
    
    return df