# Databricks notebook source
import pyspark
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import airporttime
from datetime import datetime, timedelta


# COMMAND ----------

def analyze_errors(df):

    '''
    Classifies each row from prediction dataframe into TP, FP, FN, TN and creates aggregations on various attributes 
        for analysis
    Outputs: Displays aggregate values by classification type
    '''
        
    df = df.withColumn("PRED_GROUP", F.when(((df.prediction == 1) & (df.DEP_DEL15 == 1)), "TP")
                                      .when(((df.prediction == 1) & (df.DEP_DEL15 != 1)), "FP")
                                      .when(((df.prediction == 0) & (df.DEP_DEL15 != 0)), "FN")
                                      .otherwise("TN"))
    
    grouped = df.groupBy("PRED_GROUP")
    
    temp = grouped.agg(F.mean(F.col("DISTANCE")), F.mean(F.col("CIG_CeilingHeightDim_median")), F.mean(F.col("CRS_DEP_TIME")), F.mean(F.col("VIS_Horizontal_median")), F.mean(F.col("WND_Speed_mean")))
    
    display(temp)
    
    temp = grouped.agg(F.avg(F.col("PREV_DEP_DEL15")))
    
    display(temp)


# COMMAND ----------

def analyze_errors_xgb(df):

    '''
    Classifies each row from prediction dataframe into TP, FP, FN, TN and creates aggregations on various attributes 
        for analysis. 
    Outputs: Displays aggregate values by classification type
    '''
    
    df = df.withColumn("PRED_GROUP", F.when(((df.prediction == 1) & (df.DEP_DEL15 == 1)), "TP")
                                      .when(((df.prediction == 1) & (df.DEP_DEL15 != 1)), "FP")
                                      .when(((df.prediction == 0) & (df.DEP_DEL15 != 0)), "FN")
                                      .otherwise("TN"))
    
    grouped = df.groupBy("PRED_GROUP")
    
    temp = grouped.agg(F.mean(F.col("DEP_TIME_BLK_mean_encoding")), F.mean(F.col("MONTH_Index")), F.mean(F.col("TAIL_NUM_mean_encoding")), F.mean(F.col("OP_UNIQUE_CARRIER_Index")), F.mean(F.col("ORIGIN_mean_encoding")))
    
    display(temp)
    
    temp = grouped.agg(F.avg(F.col("PREV_DEP_DEL15")))
    
    display(temp)