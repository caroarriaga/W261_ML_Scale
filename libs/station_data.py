# Databricks notebook source
import pyspark

# COMMAND ----------

def get_station_data():
    """
    This function reads in the weather stations dataset and filters relevant columns for our model.
    Inputs:
    - None
    Outputs:
    - `df_stations`: Spark DataFrame of weather station data, filtered for useful columns in our models.
    """
    df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*").cache()
    cols = ["station_id","lat","lon","neighbor_id","neighbor_name","neighbor_state","neighbor_call", "distance_to_neighbor"]
    return df_stations.select(*cols).cache()

# COMMAND ----------

