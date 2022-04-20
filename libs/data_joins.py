# Databricks notebook source
import pyspark
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import airporttime
from datetime import datetime, timedelta

# COMMAND ----------

def flight_timestamps(df, window_start=6, window_end=2):
    """
    Cleaning flight times and creating columns for the time window to grab weather data within.
    Inputs:
    - df: Spark DataFrame of flight data (all columns)
    - window_start: int. number of hours prior to flight time to start the weather data time window. Default 6 hours prior to flight time.
    - window_end: int. number of hours prior to flight time to end the weather data time window. Default 2 hours prior to flight time. Must be at least 2.
    Outputs:
    - df_airlines_dt: Spark DataFrame with cleaned flight timestamps and weather data window columns.
    """
    # parsing arguments
    window_start = int(window_start)
    window_end = int(window_end)
    if window_end < 2:
        raise Exception("window_end must be at least 2 hours prior to flight time.")
    
    # retrieving only date and time from flights
    df_airlines_dt = df.filter(df.CRS_DEP_TIME.isNotNull() & df.FL_DATE.isNotNull()) 

    # converting DEP_TIME to string, adding a leading "0" to single-digit hours, and then adding a colon in DEP_TIME for conversion to timestamp
    # also dealing with a special case where midnight is given as 24:00 instead of 00:00
    df_airlines_dt = df_airlines_dt.withColumn("CRS_DEP_TIME", df_airlines_dt["CRS_DEP_TIME"].cast("string")) \
                                    .withColumn("CRS_DEP_TIME", F.when(F.length(F.col("CRS_DEP_TIME"))<4, F.lpad(F.col("CRS_DEP_TIME"), 4, "0")).otherwise(F.col("CRS_DEP_TIME"))) \
                                    .withColumn("CRS_DEP_TIME", F.regexp_replace(F.col("CRS_DEP_TIME"), "(\d{2})(\d{2})", "$1:$2")) \
                                    .withColumn("CRS_DEP_TIME", F.when(F.col("CRS_DEP_TIME")=="24:00", "00:00").otherwise(F.col("CRS_DEP_TIME"))) \


    # joining FL_DATE and DEP_TIME and converting to timestamp to match DATE from df_weather
    df_airlines_dt = df_airlines_dt.withColumn("TIMESTAMP", F.concat(F.col("FL_DATE"), F.lit(" "), F.col("CRS_DEP_TIME"))) \
                                    .withColumn("TIMESTAMP", F.to_timestamp(F.col("TIMESTAMP"), "yyyy-MM-dd HH:mm"))
  
    # converting TIMESTAMP to UTC and calculating the acceptable window for weather observations (6 hours before departure -> 2 hours before departure)
    from datetime import datetime
    # UDF to convert a timestamp to UTC time based on the airport location
    utc_converter = udf(lambda arr: airporttime.AirportTime(arr[0]).to_utc(datetime.strptime(arr[1], "%Y-%m-%d %H:%M:%S")), TimestampType())
    df_airlines_dt = df_airlines_dt.withColumn("TIMESTAMP_UTC", utc_converter(F.array("ORIGIN", "TIMESTAMP")))
    
    # creating weather data window from UTC timestamp
    # function to create a UDF that subtracts a certain number of hours from a timestamp, to create the weather observation winow
    def weather_window(hours_before):
        return udf(lambda timestamp: timestamp - timedelta(hours=hours_before), TimestampType())
    df_airlines_dt = df_airlines_dt.withColumn("WEATHER_WINDOW_START", weather_window(window_start)("TIMESTAMP_UTC")) \
                                    .withColumn("WEATHER_WINDOW_END", weather_window(window_end)("TIMESTAMP_UTC"))
    
    return df_airlines_dt.cache()

# COMMAND ----------

def rank_closest_stations(df, n=5):
    """
    Take N closest stations for each airport.
    Inputs:
    - df: Spark DataFrame of station data
    - n: int. Rank of distance to filter by.
    Outputs:
    - df_stations_ranked: Spark DataFrame with closest n weather stations to each airport.
    """
    # parsing input
    n = int(n)
    
    # ranking and filtering closest weather stations to airports
    window = Window.partitionBy(df['station_id']).orderBy(df['distance_to_neighbor'])
    df_stations_ranked = df.select('*', F.rank().over(window).alias('dist_to_airport_rank')).filter(F.col('dist_to_airport_rank') <= n).cache()
    
    return df_stations_ranked

# COMMAND ----------

def airport_to_weather(df_airport_codes, df_stations_ranked, df_weather):
    """
    Joining airport codes -> stations -> weather
    Inputs:
    - df_airport_codes: Spark DataFrame with airport IATA codes
    - df_stations_ranked: Spark DataFrame with top n closest weather stations to each airport, from rank_closest_stations()
    - df_weather: Spark DataFrame with weather data
    Outputs:
    - airport_to_weather: Joined Spark DataFrame with airport codes -> weather stations -> weather data closest to each station
    """
    airport_to_weather = df_airport_codes.join(df_stations_ranked, airport_codes.ident == df_stations_ranked.neighbor_call, how="left").cache()
    airport_to_weather = airport_to_weather.join(df_weather, airport_to_weather.neighbor_id == df_weather.STATION, how="left").cache()
    airport_to_weather = airport_to_weather.filter(airport_to_weather.DATE.isNotNull())

    # converting DATE to UTC
    from datetime import datetime
    # UDF to convert a timestamp to UTC time based on the airport location
    utc_converter = udf(lambda arr: airporttime.AirportTime(arr[0]).to_utc(datetime.strptime(arr[1], "%Y-%m-%d %H:%M:%S")), TimestampType())
    airport_to_weather = airport_to_weather.withColumn("DATE_UTC", utc_converter(F.array("iata_code", "DATE")))
    
    return airport_to_weather.cache()

# COMMAND ----------

def full_join(df_flights, df_airport_to_weather):
    """
    Joining weather data onto flight data, where weather timestamp is within our acceptable window (defined in flight_timestamps)
    Inputs:
    - df_flights: Spark DataFrame with flight data
    - df_airport_to_weather: Joined Spark DataFrame with airport code -> weather stations -> weather data closest to each station (from airport_to_weather())
    Outputs:
    - df_join: Joined Spark DataFrame, with weather data for each flight from top n closest weather stations
    """
    # join condition: same airport, date within weather_window
    cond = [df_flights.ORIGIN == df_airport_to_weather.iata_code, df_airport_to_weather.DATE_UTC.between(df_flights.WEATHER_WINDOW_START, df_flights.WEATHER_WINDOW_END)]
    df_join = df_flights.join(df_airport_to_weather, cond, how="left").cache()
    
    return df_join

# COMMAND ----------

