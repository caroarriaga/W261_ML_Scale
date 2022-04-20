# Databricks notebook source
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.types import *

# COMMAND ----------

# PRIVATE: loading weather data from the file system
def _load_3month_weather_data():
    """
    Private function to load the three month weather data from blob storage
    Outputs: Spark DataFrame containing the three month weather data 
    """
    df = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(F.col('DATE') < "2015-04-01T00:00:00.000").cache()
    return df   

def _load_6month_weather_data():
    """
    Private function to load the six month weather data from blob storage
    Outputs: Spark DataFrame containing the six month weather data 
    """
    df = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(F.col('DATE') < "2015-07-01T00:00:00.000").cache()
    return df

def _load_full_weather_data():
    """
    Private function to load the full weather data from blob storage
    Outputs: Spark DataFrame containing the full weather data 
    """    
    df = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").cache()
    return df

# COMMAND ----------

# PRIVATE: Getting weather column constants
def _get_weather_cols():
    """
    Private function to get the weather columns to select
    Outputs: List of weather columns 
    """
    weather_select = ['STATION', 'DATE', 'SOURCE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP', 'AA1', 'AA2', 'AA3', 'AA4', 'AJ1', 'AL1', 'AL2', 'AL3', 'AW1', 'AW2', 'AW3', 'AW4']
    return weather_select

def _get_weather_col_info():
    """
    Private function to get information about each of the weather columns, to help extract the encoded data within each column
    Outputs: Dictionary of column name to the values encoded in the column and the tpyes of the values
    """
    weather_column_info = {
        'WND': {  
            'val_names': ['DirectionAngle', 'DirectionQuality', 'Type', 'Speed', 'SpeedQuality'],
            'val_types': ['int', 'string', 'string', 'int', 'string']
        },
        'CIG': {  
            'val_names': ['CeilingHeightDim', 'CeilingQuality', 'CeilingDetermination', 'CeilingAndVisibilityOK'],
            'val_types': ['int', 'string', 'string', 'string']
        },
        'VIS': {  
            'val_names': ['Horizontal', 'DistanceQuality', 'Variability', 'VariabilityQuality'],
            'val_types': ['int', 'string', 'string', 'string']
        },
        'TMP': {  
            'val_names': ['Value', 'Quality'],
            'val_types': ['int', 'string']
        },
        'DEW': {  
            'val_names': ['Value', 'Quality'],
            'val_types': ['int', 'string']
        },
        'SLP': {  
            'val_names': ['Value', 'Quality'],
            'val_types': ['int', 'string']
        },
        # Liquid precipitation (rain) has 4 possible recordings (AA1-AA4)
        'AA1': {  
            'val_names': ['RainDuration', 'RainDepth', 'RainCondition', 'RainQuality'],
            'val_types': ['int', 'int', 'string', 'string']
        },
        'AA2': {
            'val_names': ['RainDuration', 'RainDepth', 'RainCondition', 'RainQuality'],
            'val_types': ['int', 'int', 'string', 'string']
        },
        'AA3': {
            'val_names': ['RainDuration', 'RainDepth', 'RainCondition', 'RainQuality'],
            'val_types': ['int', 'int', 'string', 'string']
        },
        'AA4': {
            'val_names': ['RainDuration', 'RainDepth', 'RainCondition', 'RainQuality'],
            'val_types': ['int', 'int', 'string', 'string']
        },
        # Snow depth is the start of a snow depth data session
        'AJ1': {
            'val_names': ['SnowDepth', 'SnowDepthCondition', 'SnowDepthQuality', 'SnowEqWaterDepth', 'SnowEqWaterDepthCondition', 'SnowEqWaterDepthQuality'],
            'val_types': ['int', 'string', 'string', 'int', 'string', 'string']
        },
        # Snow accumulation has 4 recordings (AL1-AL4, but AL4 was missing) 
        'AL1': {
            'val_names': ['SnowAccumDuration', 'SnowAccumDepth', 'SnowAccumCondition', 'SnowAccumQuality'],
            'val_types': ['int', 'int', 'string', 'string']
        },
        'AL2': {
            'val_names': ['SnowAccumDuration', 'SnowAccumDepth', 'SnowAccumCondition', 'SnowAccumQuality'],
            'val_types': ['int', 'int', 'string', 'string']
        },
        'AL3': {
            'val_names': ['SnowAccumDuration', 'SnowAccumDepth', 'SnowAccumCondition', 'SnowAccumQuality'],
            'val_types': ['int', 'int', 'string', 'string']
        },
        # Present weather condition has 4 recordings (AW1-AW4)
        'AW1': {
            'val_names': ['PresentWeatherCond', 'PresentWeatherQuality'],
            'val_types': ['int', 'string']
        },
        'AW2': {
            'val_names': ['PresentWeatherCond', 'PresentWeatherQuality'],
            'val_types': ['int', 'string']
        },
        'AW3': {
            'val_names': ['PresentWeatherCond', 'PresentWeatherQuality'],
            'val_types': ['int', 'string']
        },
        'AW4': {
            'val_names': ['PresentWeatherCond', 'PresentWeatherQuality'],
            'val_types': ['int', 'string']
        },
    }
    return weather_column_info

# COMMAND ----------

# PRIVATE: Parsing and splitting weather column data
def _split_weather_cols(df, col_info):
    """
    Private function to extract the endcoded information from the weather columns
    Inputs: Spark DataFrame to split and a dictionary of the column information
    Outputs: Spark DataFrame with the encoded weather columns expanded 
    """  
    for key in col_info.keys():
        col_split = pyspark.sql.functions.split(df[key], ',')
        col_name_suffixes = col_info[key]['val_names']
        col_types = col_info[key]['val_types']
        for i in range(len(col_name_suffixes)):
            df = df.withColumn(f"{key}_{col_name_suffixes[i]}", col_split.getItem(i).cast(col_types[i]))
    return df

def _select_and_parse_weather_cols(df_weather):
    """
    Private function handle selecting and parsing the weather information
    Inputs: Spark DataFrame containing unparsed weather data
    Outputs: Spark DataFrame containing parsed weather data
    """    
    cols = _get_weather_cols()
    col_info = _get_weather_col_info()
    df_weather = df_weather.select(*cols).cache()
    df_weather = _split_weather_cols(df_weather, col_info)
    df_weather = df_weather.drop(*col_info.keys())
    return df_weather
    

# COMMAND ----------

def _drop_null_stations(df, verbose = False):
    """
    Private function to drop rows when the station ID columns is empty
    Inputs: Spark DataFrame containing weather information, Verbose = True prints out % of rows dropped
    Outputs: Spark DataFrame
    """  
    if verbose: pre_count = df.count()
    df = df.na.drop(subset=["STATION"])
    if verbose: 
        post_count = df.count()
        print(f"Dropped {pre_count - post_count} rows where STATION is null, which is {round((pre_count - post_count) / post_count * 100, 2)}% of the data")
    return df

# COMMAND ----------

# PRIVATE: Get quality code constants

def _get_bad_quality_codes():
    """
    Private function to get the bad quality codes
    Outputs: List of bad quality weather codes 
    """
    BAD_QUALITY_CODES = ['2', '3', '6', '7']
    return BAD_QUALITY_CODES

def _get_quality_col_info():
    """
    Private function to get a map of the quality column and it's corresponding value column
    Outputs: Dictionary
    """    
    QUAILTY_COLS = {
        "WND_DirectionQuality": ["WND_DirectionAngle"],
        "WND_SpeedQuality": ["WND_Speed"],
        "CIG_CeilingQuality": ["CIG_CeilingHeightDim"],
        "VIS_DistanceQuality": ["VIS_Horizontal"],
        "VIS_VariabilityQuality": ["VIS_Variability"],
        "TMP_Quality": ["TMP_Value"],
        "DEW_Quality": ["DEW_Value"],
        "SLP_Quality": ["SLP_Value"],
        "AA1_RainQuality": ["AA1_RainDuration", "AA1_RainDepth"],
        "AA2_RainQuality": ["AA2_RainDuration", "AA2_RainDepth"],
        "AA3_RainQuality": ["AA3_RainDuration", "AA3_RainDepth"],
        "AA4_RainQuality": ["AA4_RainDuration", "AA4_RainDepth"],
        "AJ1_SnowDepthQuality": ["AJ1_SnowDepth"],
        "AJ1_SnowEqWaterDepthQuality": ["AJ1_SnowEqWaterDepth"],
        "AL1_SnowAccumQuality": ["AL1_SnowAccumDuration", "AL1_SnowAccumDepth"],
        "AL2_SnowAccumQuality": ["AL2_SnowAccumDuration", "AL2_SnowAccumDepth"],
        "AL3_SnowAccumQuality": ["AL3_SnowAccumDuration", "AL3_SnowAccumDepth"],
        "AW1_PresentWeatherQuality": ["AW1_PresentWeatherCond"],
        "AW2_PresentWeatherQuality": ["AW2_PresentWeatherCond"],
        "AW3_PresentWeatherQuality": ["AW3_PresentWeatherCond"],
        "AW4_PresentWeatherQuality": ["AW4_PresentWeatherCond"],
       } 
    return QUAILTY_COLS

# COMMAND ----------

# PRIVATE: Handling data for bad quality codes 
def _remap_quality_cols(df):
    """
    Private function to remap quality codes to either good 1 or bad 0
    Inputs: Spark DataFrame containing weather data 
    Outputs: Spark DataFrame
    """  
    quality_cols = _get_quality_col_info()
    bad_quality_codes = _get_bad_quality_codes()
    for col in quality_cols.keys():
        df = df.withColumn(col, F.when(df[col].isNull(), F.lit(None)) \
                                .when(df[col].isin(bad_quality_codes), 0) \
                                .otherwise(1))
    return df

def _nullify_bad_quality_entries(df):
    """
    Private function to nullify values with bad quality codes 
    Inputs: Spark DataFrame containing weather data
    Outputs: Spark DataFrame 
    """
    quality_cols = _get_quality_col_info()
    for quality_col in quality_cols.keys():
        for value_col in quality_cols[quality_col]:
            df = df.withColumn(value_col, F.when(df[quality_col] == 0, F.lit(None)).otherwise(df[value_col]))
    return df

def _handle_bad_quality(df):
    """
    Private function to remap quality codes to good or bad, then nullify corresponding values with bad quality codes 
    Inputs: Spark DataFrame containing weather data
    Outputs: Spark DataFrame
    """    
    df = _remap_quality_cols(df)
    df = _nullify_bad_quality_entries(df)
    return df

# COMMAND ----------

# PRiVATE: Get constants for missing values for each column    
def _get_missing_value_codes():
    """
    Private function to get a dictionary of a column and the value that represents missing
    Outputs: Dictionary of column names and their missing value code 
    """  
    MISSING_VALUE_CODES = {
        "WND_DirectionAngle": "999",
        "WND_Type": "9",
        "WND_Speed": "9999",
        "CIG_CeilingHeightDim": "99999",
        "CIG_CeilingDetermination": "9",
        "CIG_CeilingAndVisibilityOK": "9",
        "VIS_Horizontal": "999999",
        "VIS_Variability": "9",
        "TMP_Value": "9999",
        "DEW_Value": "9999",
        "SLP_Value": "99999",
        "AA1_RainDuration": "99",
        "AA2_RainDuration": "99",
        "AA3_RainDuration": "99",
        "AA4_RainDuration": "99",
        "AA1_RainDepth": "9999",
        "AA2_RainDepth": "9999",
        "AA3_RainDepth": "9999",
        "AA4_RainDepth": "9999",
        "AJ1_SnowDepth": "9999",
        "AJ1_SnowEqWaterDepth": "999999",
        "AL1_SnowAccumDuration": "99",
        "AL2_SnowAccumDuration": "99",
        "AL3_SnowAccumDuration": "99",
        "AL1_SnowAccumDepth": "999",
        "AL2_SnowAccumDepth": "999",
        "AL3_SnowAccumDepth": "999",
    } 
    return MISSING_VALUE_CODES

# COMMAND ----------

# PRIVATE: Handling missing values 

# Special case for WND_Type, if this is marked as '9' for missing, but is has a value of '0000' for WND_Speed, then it means calm winds. We can replace the WND_Type with 'C', which is the code for calm winds.
def _handle_wind_special_case(df):
    """
    Private function to handle special entires for calm wind 
    Inputs: Spark DataFrame containing weather data
    Outputs: Spark DataFrame
    """
    df = df.withColumn('WND_Type', F.when((df['WND_Type'] == '9') & (df['WND_Speed'] == '9999'), 'C').otherwise(df['WND_Type']))
    return df

def _convert_missing_to_null(df):
    """
    Private function to nullify values when their entry is a missing code
    Inputs: Spark DataFrame containing weather data
    Outputs: Spark DataFrame
    """
    missing_value_codes = _get_missing_value_codes()
    for col in missing_value_codes.keys():
        df = df.withColumn(col, F.when(df[col] == missing_value_codes[col], F.lit(None)).otherwise(df[col]))
    return df

def _handle_missing_values(df):
    """
    Private function to handle entries with a missing code value
    Inputs: Spark DataFrame containing weather data
    Outputs: Spark DataFrame
    """    
    df = _handle_wind_special_case(df)
    df = _convert_missing_to_null(df)
    return df

# COMMAND ----------

# PRIVATE: Finding the average across repeated columns, not including the column if it's null
def _average_repeated_columns(df):
    """
    Private function to combine repeated columns by finding their average
    Inputs: Spark DataFrame containing weather data
    Outputs: Spark DataFrame
    """
    concat_non_null = udf(lambda cols: [x for x in cols if x != None])
    average_non_null = udf(lambda vals: sum(x for x in vals) / len(vals) if sum(x for x in vals) > 0 else None, IntegerType())

    df = df.withColumn('AA_RainDepth', average_non_null(concat_non_null(F.array(F.col('AA1_RainDepth'), F.col('AA2_RainDepth'), F.col('AA3_RainDepth'), F.col('AA4_RainDepth')))))
    df = df.withColumn('AL_SnowAccumDepth', average_non_null(concat_non_null(F.array(F.col('AL1_SnowAccumDepth'), F.col('AL2_SnowAccumDepth')))))
    df = df.withColumn('AA_RainDuration', average_non_null(concat_non_null(F.array(F.col('AA1_RainDuration'), F.col('AA2_RainDuration'), F.col('AA3_RainDuration'), F.col('AA4_RainDuration')))))
    df = df.withColumn('AL_SnowAccumDuration', average_non_null(concat_non_null(F.array(F.col('AL1_SnowAccumDuration'), F.col('AL2_SnowAccumDuration')))))
    
    df = df.drop(*['AA1_RainDepth', 'AA2_RainDepth', 'AA3_RainDepth', 'AA4_RainDepth', 'AL1_SnowAccumDepth', 'AL2_SnowAccumDepth', 
                  'AA1_RainDuration', 'AA2_RainDuration', 'AA3_RainDuration', 'AA4_RainDuration', 'AL1_SnowAccumDuration', 'AL2_SnowAccumDuration'])
    
    return df

# COMMAND ----------

# PRIVATE: Drop duplicate rows
def _drop_duplicates_weather(df, verbose = False):
    """
    Private function to drop duplicate weather rows
    Inputs: Spark DataFrame containing weather data
    Outputs: Spark DataFrame
    """  
    duplicate_cols = ['STATION', 'DATE', 'SOURCE', 'REPORT_TYPE']
    if verbose: pre_count = df.count()
    df = df.dropDuplicates(duplicate_cols)
    if verbose: 
        post_count = df.count()
        print(f"Dropped {pre_count - post_count} duplicate rows, which is {round((pre_count - post_count) / post_count * 100, 2)}% of the data")
    return df
    

# COMMAND ----------

# PRIVATE: Putting it all together
def _parse_and_clean_weather_data(df, verbose = False):
    """
    Private function to parse and clean weather data
    Inputs: Spark DataFrame containing weather data
    Outputs: Spark DataFrame
    """  
    df = _select_and_parse_weather_cols(df)
    df = _drop_null_stations(df, verbose = verbose)
    df = _handle_bad_quality(df)
    df = _handle_missing_values(df)
    df = _drop_duplicates_weather(df, verbose = verbose)
    df = _average_repeated_columns(df)
    return df

# COMMAND ----------

# PUBLIC: Load and clean weather data

def get_3month_weather_data(verbose = False):
    """
    Get the parsed three month weather data
    Outputs: Spark DataFrame
    """  
    df = _load_3month_weather_data()
    df = _parse_and_clean_weather_data(df, verbose = verbose)
    return df

def get_6month_weather_data(verbose = False):
    """
    Get the parsed six month weather data
    Outputs: Spark DataFrame
    """  
    df = _load_6month_weather_data()
    df = _parse_and_clean_weather_data(df, verbose = verbose)
    return df

def get_full_weather_data(verbose = False):
    """
    Get the parsed full weather data
    Outputs: Spark DataFrame
    """  
    df = _load_full_weather_data()
    df = _parse_and_clean_weather_data(df, verbose = verbose)
    return df