# Databricks notebook source
def get_airport_codes_full():
    airport_codes = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/airport_codes_csv.csv")
    codes_cols = ['ident', 'elevation_ft', 'iata_code', 'coordinates']
    airport_codes = airport_codes.select(*codes_cols).filter(airport_codes.iata_code != 'null').cache()
    df_airlines_2015 = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2015.parquet/*").cache()
    df_airlines_2016 = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2016.parquet/*").cache()
    df_airlines_2017 = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2017.parquet/*").cache()
    df_airlines_2018 = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2018.parquet/*").cache()
    df_airlines_train = df_airlines_2015.unionByName(df_airlines_2016, allowMissingColumns = True).unionByName(df_airlines_2017, allowMissingColumns = True).unionByName(df_airlines_2018, allowMissingColumns = True).cache()
    df_airlines_dist = df_airlines_train.select('ORIGIN').distinct().cache()
        #import airport codes, map ident on iata code to origin and neighbor_call
    airport_codes = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/airport_codes_csv.csv")
    codes_cols = ['ident', 'elevation_ft', 'iata_code', 'coordinates']
    airport_codes = airport_codes.select(*codes_cols).filter(airport_codes.iata_code != 'null').cache()
    # display(airport_codes)
    airport_codes_full = df_airlines_dist.join(airport_codes, airport_codes.iata_code == df_airlines_dist.ORIGIN, how = 'left')
    return airport_codes_full.select('iata_code', 'ident', 'elevation_ft', 'coordinates').cache()