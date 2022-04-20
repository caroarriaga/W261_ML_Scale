# Databricks notebook source
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import *
from datetime import datetime

# COMMAND ----------

# PRIVATE: Doing the weather aggregation for numeric columns
def _do_numeric_aggregation(values, distances, flight_times, weather_times):
    """
    Private function to aggregate rows of weather columns containing numeric values by calculating the weighted average
    Inputs:
    - values: Spark SQL array of the numeric values in each row
    - distances: Spark SQL array of the corresponsing distances from the flight's airport
    - flight_times: Spark SQL array of the flight time (repeated values)
    - weather_times: Spark SQL array of the corresponding weather report times
    Outputs: Float of the weighted average
    """  
    numerator = None
    denominator = None
    for i in range(len(values)):
        if distances[i] is not None:
            # Calculating difference between flight planned take off and weather report time (in minutes)
            time_diff = (flight_times[i] - weather_times[i]) / 60
            # handling case when weather station is at the airport so distance is 0
            if distances[i] == 0:
                inverted_distance = 1
            else: 
                inverted_distance = 1/distances[i]
            # it should never be the case that time_diff is 0 as we take weather reports from 2 hours prior to take off, but adding as a safeguard 
            if time_diff == 0:
                inverted_time = 1
            else:
                inverted_time = 1 / time_diff
            if numerator is None:
                numerator = 0
            if denominator is None:
                denominator = 0
            numerator += values[i] * inverted_distance * inverted_time
            denominator += inverted_distance * inverted_time
    if numerator is not None and denominator is not None:
        weighted_avg = round(numerator / denominator, 2)
    else: 
        weighted_avg = None
    return weighted_avg

def _do_categoric_aggregation(values):
    """
    Private function to aggregate rows of weather columns containing categorical values by finding the mode
    Inputs:
    - values: Spark SQL array of the categorical values in each row

    Outputs: String of the aggregated category
    """      
    # filter out null values
    values = list(filter(None, values))
    # return the most common category
    counts = {}
    if len(values):
        return max(set(values), key = values.count)     
    else:
        return None
    
def _aggregate_weather_reports(df):
    """
    Private function to aggregate rows of weather columns for a flight
    Inputs:
    - df: Spark DataFrame containing joined weather and flight data
    Outputs: Spark DataFrame
    """      
    _agg_weather_numeric_udf = F.udf(_do_numeric_aggregation, FloatType())
    _agg_weather_categorical_udf = F.udf(_do_categoric_aggregation, StringType())
    
    # Temp adding this here until the saved join has been re-ran
#     df = df.withColumn('AA_RainDepth', F.col('AA_RainDepth').cast(IntegerType())) \
#                             .withColumn('AL_SnowAccumDepth', F.col('AL_SnowAccumDepth').cast(IntegerType())) \
#                             .withColumn('AA_RainDuration', F.col('AA_RainDuration').cast(IntegerType())) \
#                             .withColumn('AL_SnowAccumDuration', F.col('AL_SnowAccumDuration').cast(IntegerType())) 
    
    flight_info = ['DEP_DEL15','CRS_DEP_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'ARR_DEL15', 
               'ARR_DELAY', 'ARR_DELAY_GROUP', 'ARR_DELAY_NEW',
               'ARR_TIME',  'CRS_ELAPSED_TIME', 
               'DEP_DELAY', 'DEP_DELAY_GROUP', 'DEP_DELAY_NEW', 
               'DEP_TIME', 'DEP_TIME_BLK', 'DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID',
                 'DEST_CITY_NAME','DEST_STATE_FIPS',
                 'DEST_STATE_NM','DEST_WAC','OP_CARRIER_AIRLINE_ID',
                 'OP_UNIQUE_CARRIER','ORIGIN_AIRPORT_ID',
                 'ORIGIN_AIRPORT_SEQ_ID','ORIGIN_STATE_FIPS', 
                 'ORIGIN_STATE_NM', 'ORIGIN_WAC','ORIGIN_CITY_NAME', 
                 'DAY_OF_WEEK',  'DISTANCE', 'DISTANCE_GROUP', 'ORIGIN_STATE_ABR']

    date_time = ['FL_DATE', 
                 'WEATHER_WINDOW_START',
                 'WEATHER_WINDOW_END' ,
                 'TIMESTAMP_UTC']

    cause_delay = ['NAS_DELAY','OP_CARRIER','SECURITY_DELAY','CARRIER_DELAY']

    identifiers = ['OP_CARRIER_FL_NUM','TAIL_NUM','TIMESTAMP', 'YEAR','MONTH','DAY_OF_MONTH']

    location = ['ORIGIN','DEST', 'ORIGIN_CITY_MARKET_ID','DEST_CITY_MARKET_ID', 'DEST_STATE_ABR']
    
    groupBy_cols = flight_info + date_time + cause_delay + identifiers + location
    
    df = df.groupBy(*groupBy_cols) \
        .agg(_agg_weather_numeric_udf(F.collect_list(col('TMP_Value')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('TMP_Value'), \
             _agg_weather_numeric_udf(F.collect_list(col('DEW_Value')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('DEW_Value'), \
             _agg_weather_numeric_udf(F.collect_list(col('VIS_Horizontal')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('VIS_Horizontal'), \
             _agg_weather_numeric_udf(F.collect_list(col('SLP_Value')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('SLP_Value'), \
             _agg_weather_numeric_udf(F.collect_list(col('AA_RainDepth')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('AA_RainDepth'), \
             _agg_weather_numeric_udf(F.collect_list(col('AA_RainDuration')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('AA_RainDuration'), \
             _agg_weather_numeric_udf(F.collect_list(col('AL_SnowAccumDuration')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('AL_SnowAccumDuration'), \
             _agg_weather_numeric_udf(F.collect_list(col('AL_SnowAccumDepth')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('AL_SnowAccumDepth'), \
             _agg_weather_numeric_udf(F.collect_list(col('AJ1_SnowDepth')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('AJ1_SnowDepth'), \
             _agg_weather_numeric_udf(F.collect_list(col('AJ1_SnowEqWaterDepth')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('AJ1_SnowEqWaterDepth'), \
             _agg_weather_numeric_udf(F.collect_list(col('WND_Speed')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('WND_Speed'), \
             _agg_weather_numeric_udf(F.collect_list(col('WND_DirectionAngle')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('WND_DirectionAngle'), \
             _agg_weather_numeric_udf(F.collect_list(col('CIG_CeilingHeightDim')), 
                                      F.collect_list(col('distance_to_neighbor')), 
                                      F.collect_list(col('TIMESTAMP_UTC').cast('long')), 
                                      F.collect_list(col('DATE_UTC').cast('long')) ) \
             .alias('CIG_CeilingHeightDim'),
             _agg_weather_categorical_udf(F.collect_list(col('AW1_PresentWeatherCond'))).alias('AW1_PresentWeatherCond'),
             _agg_weather_categorical_udf(F.collect_list(col('AW2_PresentWeatherCond'))).alias('AW2_PresentWeatherCond'),
             _agg_weather_categorical_udf(F.collect_list(col('AW3_PresentWeatherCond'))).alias('AW3_PresentWeatherCond'),
             _agg_weather_categorical_udf(F.collect_list(col('AW4_PresentWeatherCond'))).alias('AW4_PresentWeatherCond'),
             _agg_weather_categorical_udf(F.collect_list(col('VIS_Variability'))).alias('VIS_Variability'),
             _agg_weather_categorical_udf(F.collect_list(col('WND_Type'))).alias('WND_Type')).cache()

    
    return df


# COMMAND ----------

# PUBLIC: Aggregate weather reports in the joined dataset
def aggregate_weather_reports(df):
    """
    Takes the joined, unaggregated DF and performs an aggregation of the weather reports for each flight
    Inputs: 
    - df: Spark DataFrame containing joined flights and weather data  
    Outputs: Spark DataFrame
    """
    df = _aggregate_weather_reports(df)
    return df