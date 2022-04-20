# Databricks notebook source
import pyspark
from pyspark.sql.types import StringType, BooleanType, IntegerType
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.ml.feature import Interaction
from itertools import combinations

# COMMAND ----------

def exclude_features(df):
    '''
    Removes unnecesary columns from df based on EDA. 
    input: dataframe
    output: dataframe without unnecessary features features
    '''
    
    # features that carry leakage - exclude
    leakage = ['ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'ARR_DEL15', 
               'ARR_DELAY', 'ARR_DELAY_GROUP', 'ARR_DELAY_NEW',
               'ARR_TIME',  'CRS_ELAPSED_TIME', 
               'DEP_DELAY', 'DEP_DELAY_GROUP', 'DEP_DELAY_NEW', 
               'DEP_TIME', 'DEP_TIME_BLK']

    # features related to datetime needed to exclude from regression
    date_time = ['FL_DATE', 
                 'WEATHER_WINDOW_START',
                 'WEATHER_WINDOW_END' ]

# These are already removed after weather aggregation group by - commenting out 
#     # features used to extract weather data - exclude
#     stations = ['coordinates', 'station_id','lat','lon',
#                 'neighbor_id','neighbor_name','neighbor_state',
#                 'neighbor_call',
#                 'dist_to_airport_rank','STATION','DATE', 
#                 'SOURCE', 'LATITUDE', 'LONGITUDE', 
#                 'ELEVATION', 'NAME','CALL_SIGN']

    # duplicate feature information - exclude
    duplicate = ['DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID',
                 'DEST_CITY_NAME','DEST_STATE_FIPS',
                 'DEST_STATE_NM','DEST_WAC','OP_CARRIER_AIRLINE_ID',
                 'OP_UNIQUE_CARRIER','ORIGIN_AIRPORT_ID',
                 'ORIGIN_AIRPORT_SEQ_ID','ORIGIN_STATE_FIPS', 
                 'ORIGIN_STATE_NM', 'ORIGIN_WAC','ident','ORIGIN_CITY_NAME']

    # cause of delay is also information leakage - exclude
    cause_delay = ['NAS_DELAY','OP_CARRIER','SECURITY_DELAY','CARRIER_DELAY']

    # can be used to get new features - exclude
    #identifiers = ['OP_CARRIER_FL_NUM','TAIL_NUM','iata_code','TIMESTAMP', 'YEAR','MONTH','DAY_OF_MONTH']
    identifiers = ['OP_CARRIER_FL_NUM','TAIL_NUM','TIMESTAMP', 'YEAR','MONTH','DAY_OF_MONTH']

    # features related to origin and destination
    location = ['ORIGIN','DEST', 'ORIGIN_CITY_MARKET_ID','DEST_CITY_MARKET_ID', 'DEST_STATE_ABR']

    # featuresToExclude = leakage + date_time + stations + duplicate + cause_delay 
    featuresToExclude = leakage + date_time + duplicate + cause_delay 
    
    return df.drop(*featuresToExclude)

# COMMAND ----------

def exclude_constant_value_features(df):
    '''
    Removes features that don't change value and have no missing values.
    Features are hardcoded based on EDA.
    Input: dataframe
    Output: dataframe without excluded features
    '''

#     constant_value_features = ['WND_SpeedQuality','VIS_DistanceQuality','DEW_Quality',
#                                'TMP_Quality','WND_DirectionQuality','CARRIER_DELAY',
#                                'VIS_VariabilityQuality', 'REPORT_TYPE']
    constant_value_features = ['CARRIER_DELAY']
    
    return df.drop(*constant_value_features)

# COMMAND ----------

# this is not required after aggreation
def exclude_extreme_missing_values_features(df):
    '''
    missing values contain over 99% missing data
    *CIG_CeilingAndVisibilityOK it's mainly 'N' meaning status wasn't reported
    *CIG_CeilingDetermination method used to create the report, most is M: Measured
    *AA1_RainCondition methos used to measue, most is missing, second is 'trace'
    '''
    missing_values = ['WND_SpeedQuality','AL1_SnowAccumQuality',
                      'AA4_RainQuality'  ,'AL2_SnowAccumQuality',
                      'AL3_SnowAccumDuration','AL3_SnowAccumDepth',
                      'AW4_PresentWeatherQuality','AL2_SnowAccumCondition',
                      'AA4_RainCondition','AJ1_SnowEqWaterDepthCondition',
                      'AJ1_SnowDepthCondition','CIG_CeilingAndVisibilityOK',
                      'AL3_SnowAccumCondition',
                      'CIG_CeilingDetermination' ,'AA1_RainCondition',
                      'AA3_RainCondition', 'AA2_RainCondition','AL1_SnowAccumCondition']
    return df.drop(*missing_values)

# COMMAND ----------

def exclude_extreme_missing_values_features_custom(df, list_features_exclude):
    return df.drop(*list_features_exclude)

# COMMAND ----------

def fill_nulls_with_zero(df):
    '''
    Fills hard coded columns nulls with zeros.
    Input: Dataframe
    Output: Updated Dataframe
    
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
#     fillWithZero = ['AL3_SnowAccumQuality','CIG_CeilingQuality','AL_SnowAccumDepth'
#                     ,'SLP_Quality', 'AJ1_SnowEqWaterDepth','AJ1_SnowEqWaterDepthQuality'
#                     ,'AA2_RainQuality' ,'AA1_RainQuality','AJ1_SnowDepthQuality'
#                     ,'AA_RainDuration','VIS_Variability','AA_RainDepth','AA3_RainQuality']
    fillWithZero = ['AL_SnowAccumDepth', 'AJ1_SnowEqWaterDepth','AA_RainDuration','VIS_Variability','AA_RainDepth']
    
    return df.fillna(0, subset=fillWithZero)

# COMMAND ----------

def fill_nulls_with_zero_custom(df, list_of_features):
    '''
    Fills null values on custom list of features with zeros.
    Input: Dataframe, list of features
    Output: Updated Dataframe
    '''

    return df.fillna(0, subset=list_of_features)

# COMMAND ----------

def cast_features_to_integers(df, list_of_features):
    '''
    Casts selected features into integers
    Input: Dataframe, list of features
    Output: Updated dataframe
    '''
    
    for feature in list_of_features:
        if feature == "CRS_DEP_TIME":
            # CRS_DEP_TIME is formatted differently, need regex before converting to int
            df = df.withColumn("CRS_DEP_TIME",(F.regexp_replace(col("CRS_DEP_TIME"), "[:]","")).cast(IntegerType()))
        else:
            df=df.withColumn(f"{feature}",col(f"{feature}").cast(IntegerType()))
    return df

# COMMAND ----------

def cast_features_to_strings(df, list_of_features):
    '''
    Casts selected features into strings
    Input: Dataframe, list of features
    Output: Updated dataframe
    '''
    for feature in list_of_features:
        df=df.withColumn(f"{feature}",col(f"{feature}").cast(StringType()))
    return df

# COMMAND ----------

def fill_null_values_with_group_mean(df, fillWithMean, identifiers):
    '''
    Replaces null values in selected columns with group mean of identifiers
    input: dataframe, list with features to add mean values, group identifiers
    output: dataframe with null values filled with mean
    '''
    w = Window.partitionBy([F.col(x) for x in identifiers])
    for feature in fillWithMean:
        df = df.withColumn(f"{feature}_mean", 
                           F.when(F.col(feature).isNull(),
                                F.avg(df[feature]).over(w)) \
                           .otherwise(df[feature])).drop(feature)
    return df

# COMMAND ----------

def fill_null_values_with_group_median(df, fillWithMedian,identifiers):
    '''
    Replaces null values in selected columns with group median of identifiers
    input: dataframe, list with features to add median values, group identifiers
    output: dataframe with null values filled with median
    '''
    w = Window.partitionBy([F.col(x) for x in identifiers])
    for feature in fillWithMedian:
        df = df.withColumn(f"{feature}_median", 
                           F.when(F.col(feature).isNull(),
                                F.percentile_approx(df[feature],0.5).over(w)) \
                           .otherwise(df[feature])).drop(feature)
    return df

# COMMAND ----------

def fill_null_values_with_group_mode(df, fillWithMode,identifiers, fallbackValue):
    '''
    Replaces null values in selected columns with group mode of identifiers
    input: dataframe, list with features to add mode values, group identifiers
    output: dataframe with null values filled with mode
    '''
    def find_group_mode(values):
        # filter out null values
        values = list(filter(None, values))
        # return the most common category
        counts = {}
        if len(values):
            return max(set(values), key = values.count)     
        else:
            return fallbackValue
    find_group_mode_udf = F.udf(find_group_mode)
    w = Window.partitionBy([F.col(x) for x in identifiers])
    for feature in fillWithMode:
        df = df.withColumn(f"{feature}_mode", 
                           F.when(F.col(feature).isNull(),
                                find_group_mode_udf(F.collect_list(F.col(feature))).over(w)) \
                           .otherwise(df[feature])).drop(feature)
    return df

# COMMAND ----------

def merge_two_col_null_values(df, col1, col2):
    '''
    Finds null values in col1 and replaces with values in col2 when not null.
    Input: dataframe, column 1 name (string), column 2 name (string) 
    output: dataframe, merge column name (string)
    '''
    
    merged_df= df.withColumn(f"{col1}_merged", 
                    F.when(F.col(f'{col1}').isNull(), 
                         df[f'{col2}']) \
                    .otherwise(df[f'{col1}'])) \
          .drop(*[col1,col2])
    
    return merged_df, f"{col1}_merged"

# COMMAND ----------

def get_transformed_df(df_joined):
    '''
    Gets the joined dataframe and transforms features for training.
    Input: Dataframe after joined
    Output: Dataframe with transformations: fill missing values, cast to specific types, merge columes, exluded features
    '''
    
    # Rajiv - commented 2 lines below
    
    # drop unecessary features
    # df_joined = exclude_features(df_joined)

    # Remove constant value variables
    # df_joined = exclude_constant_value_features(df_joined)

    # remove features missing 99% of the values
    # df_joined = exclude_extreme_missing_values_features(df_joined)

    # fill selected features null values with zero
    df_joined = fill_nulls_with_zero(df_joined)
    
    # Features that help identify flights
    identifiers = ['OP_CARRIER_FL_NUM','TAIL_NUM','MONTH','TIMESTAMP', 'DAY_OF_MONTH']

    # Fill median and mean values
    fillNullsWithMedian =['CIG_CeilingHeightDim','VIS_Horizontal','WND_DirectionAngle','DEW_Value']
    fillNullsWithMean = ['WND_Speed','SLP_Value', 'AL_SnowAccumDuration', 'AJ1_SnowDepth'] #,'TMP_Value' removed
    fillNullsWithMode = ['weather_condition']
    df_joined = fill_null_values_with_group_mean(df_joined, fillNullsWithMean, identifiers[:2])
    df_joined = fill_null_values_with_group_median(df_joined, fillNullsWithMedian, identifiers[:2])
    
    # Cast columns to integers
    features_to_integers=['CRS_DEP_TIME','AL_SnowAccumDepth']
    df_joined=cast_features_to_integers(df_joined, features_to_integers)
    
    # Merge weather conditions into a single column when they carry additional information
    df_joined, merged_name = merge_two_col_null_values(df_joined, 'AW1_PresentWeatherCond', 'AW2_PresentWeatherCond')
    df_joined, merged_name = merge_two_col_null_values(df_joined, merged_name, 'AW3_PresentWeatherCond')
    df_joined, merged_name = merge_two_col_null_values(df_joined, merged_name, 'AW4_PresentWeatherCond')
    df_joined=df_joined.withColumnRenamed(merged_name,"weather_condition")
    
    #df_joined = fill_null_values_with_group_mode(df_joined, fillNullsWithMode, identifiers[:2], '00')
    
    # if there were nulls within the groups for mean, median or merged columns, fill them with 0 if apropiate
    fillWithZero=['weather_condition','WND_DirectionAngle_median','AA_RainDepth',
                  'AJ1_SnowDepth_mean','AL_SnowAccumDuration_mean','WND_Speed_mean','VIS_Horizontal_median']
    df_joined = fill_nulls_with_zero_custom(df_joined,fillWithZero)
    
    # cast as string weather condition (each number represents a different condition)
    string_features = ['weather_condition','MONTH','DAY_OF_MONTH', 'DAY_OF_WEEK']
    df_joined = cast_features_to_strings(df_joined,string_features)
    
    # remove unecessary features, a bit more of cleaning
    #features_to_drop = ['AA2_RainQuality', 'AA3_RainQuality','AW2_PresentWeatherQuality','AW3_PresentWeatherQuality']
    #df_joined = exclude_extreme_missing_values_features_custom(df_joined,features_to_drop)
    
    return df_joined

# COMMAND ----------


def add_interactions(df, features):
    '''
    Adds interaction terms between features
    Input: Dataframe
    Output: Dataframe with new cols for each combination between two features
    '''
#     features = df.columns
#     features.remove('DEP_DEL15')
    interaction_cols = []
    # Get all permutations of features
    comb = combinations(features, 2)
    
    for c in comb:
#         print(c)
        interaction=Interaction()
        interaction.setInputCols([f'{c[0]}',f'{c[1]}'])
        interaction.setOutputCol(f'{c[0]}_{c[1]}')
        df = interaction.transform(df)
        interaction_cols.append(f'{c[0]}_{c[1]}')
    return df, interaction_cols

# Test
# df = spark.createDataFrame([(0.0, 1.0), (2.0, 3.0)], ["a", "b"])
# df = add_interactions(df)
# display(df)

# COMMAND ----------

def add_holidays(df_train, df1):
    '''
    Adds holiday importance and holiday binary column to a dataset
    Input: Dataframe, Holidays table
    Ouput: Updated dataframe
    '''

    # # change date to correct format
    for column in ["Month", "Day"]:
        df1 = df1.withColumn(column, F.when(F.length(F.col(column))<2, F.lpad(F.col(column), 2, "0")).otherwise(F.col(column)))
    df1 = df1.withColumn("Date", F.concat(col("Year"), F.lit("-"), col("Month"), F.lit("-"), F.col("Day")))
    df1 = df1.withColumn("Date_time", F.to_timestamp(df1.Date, "yyyy-mm-dd"))
    # df1 = df1.withColumn("Is_Holiday", 1).select('Date', 'Holiday', 'Date_time', 'Importance', 'Is_Holiday')

    df_holidays_train = df_train.join(df1, df_train.FL_DATE == df1.Date, how = 'left').cache()
    
    df_holidays_train=df_holidays_train.withColumn("is_holiday", F.when(df_holidays_train.Importance >=0, 1).otherwise(0))
    df_holidays_train=df_holidays_train.withColumn("holiday_imp", F.when(df_holidays_train.Importance ==0, 1) \
                                               .when(df_holidays_train.Importance ==1, 2) \
                                               .when(df_holidays_train.Importance ==2, 3) \
                                               .otherwise(0))
    df_holidays_train=df_holidays_train.drop(*['Date','Holiday','WeekDay','Month','Day','Year','Importance','Date_time'])
    
    return df_holidays_train

# COMMAND ----------

def add_prev_del_ori_dest_proba(df, df_del):
    '''
    Adds the probability column of a previous delay given a specific Origin-Destiny grouped by carrier.
    Input: df to add new column, df with all previous probabilities
    Output: df with previous delayed probability column
    '''

    cond = [df.ORIGIN==df_del.src, 
            df.DEST==df_del.dst,
            df.OP_UNIQUE_CARRIER==df_del.carrier]

    df = df.join(df_del, cond, how = 'left').drop(*['src','dst','carrier'])
    return df

# COMMAND ----------

def add_in_degree_origin(df,df1):
    '''
    Adds the in degree column based on the origin of a flight
    Input: dataframe to add in degree, InDegree table
    Output: dataframe with in degree column
    '''
    df = df.join(df1, 
                 df.ORIGIN==df1.id,
                 how = 'left').drop('id')
    return df

# COMMAND ----------

def add_balancing_ratio(df):
    '''
    Adds the balancing ratio -positives divided by total samples- as a column.
    Input: dataframe
    Output: Updated dataframe
    '''
    balancingRatio = (df.where(df.DEP_DEL15 == 0).count()) / (df.count())
    df = df.withColumn("classWeights", F.when(df.DEP_DEL15 == 1,balancingRatio).otherwise(1-balancingRatio))
    return df