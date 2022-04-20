# Databricks notebook source
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.sql.types import *
from datetime import datetime
import databricks.koalas as ks
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

blob_container = "w261-scrr" # The name of your container created in https://portal.azure.com
storage_account = "midsw261rv" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261scrr" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261scrrkey" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

df_joined = spark.read.parquet(f"{blob_url}/join_full_0329").filter(col("YEAR") == '2015').cache()
#display(df_joined)

# COMMAND ----------

display(df_joined)

# COMMAND ----------

df_joined.count()

# COMMAND ----------

# MAGIC %run "./libs/weather_aggregation"

# COMMAND ----------

df_aggregated = aggregate_weather_reports(df_joined).cache()
#df_aggregated_6 = aggregate_weather_reports(df_joined_6).cache()

# COMMAND ----------

display(df_aggregated.limit(10))

# COMMAND ----------

df_aggregated.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assessing relationships between numerical weather features and outcome variable 
# MAGIC Getting a rough sense of correlation between numerical weather features and outcome variable using mean and stdv:

# COMMAND ----------

display(df_aggregated.groupBy('DEP_DEL15').agg(
    F.mean(col('TMP_Value') / 10).alias('TMP_avg'), F.stddev(col('TMP_Value') / 10).alias('TMP_stdv'), \
    F.mean(col('DEW_Value')).alias('DEW_avg'), F.stddev(col('DEW_Value')).alias('DEW_stdv'), \
    F.mean(col('VIS_Horizontal')).alias('VIS_Horizontal_avg'), F.stddev(col('VIS_Horizontal')).alias('VIS_Horizontal_stdv'), \
    F.mean(col('WND_Speed')).alias('WND_Speed_avg'), F.stddev(col('WND_Speed')).alias('WND_Speed_stdv'), \
    F.mean(col('SLP_Value')).alias('SLP_Value_avg'), F.stddev(col('SLP_Value')).alias('SLP_Value_stdv'), \
    F.mean(col('AA_RainDepth')).alias('AA_RainDepth_avg'), F.stddev(col('AA_RainDepth')).alias('AA_RainDepth_stdv'), \
    F.mean(col('AA_RainDuration')).alias('AA_RainDuration_avg'), F.stddev(col('AA_RainDuration')).alias('AA_RainDuration_stdv'), \
    F.mean(col('AL_SnowAccumDuration')).alias('AL_SnowAccumDuration_avg'), F.stddev(col('AL_SnowAccumDuration')).alias('AL_SnowAccumDuration_stdv'), \
    F.mean(col('AL_SnowAccumDepth')).alias('AL_SnowAccumDepth_avg'), F.stddev(col('AL_SnowAccumDepth')).alias('AL_SnowAccumDepth_stdv'), \
    F.mean(col('AJ1_SnowDepth')).alias('AJ1_SnowDepth_avg'), F.stddev(col('AJ1_SnowDepth')).alias('AJ1_SnowDepth_stdv'), \
    F.mean(col('AJ1_SnowEqWaterDepth')).alias('AJ1_SnowEqWaterDepth_avg'), F.stddev(col('AJ1_SnowEqWaterDepth')).alias('AJ1_SnowEqWaterDepth_stdv'), \
    F.mean(col('WND_Speed')).alias('WND_Speed_avg'), F.stddev(col('WND_Speed')).alias('WND_Speed_stdv'), \
    F.mean(col('WND_DirectionAngle')).alias('WND_DirectionAngle_avg'), F.stddev(col('WND_DirectionAngle')).alias('WND_DirectionAngle_stdv'), \
    F.mean(col('CIG_CeilingHeightDim')).alias('CIG_CeilingHeightDim_avg'), F.stddev(col('CIG_CeilingHeightDim')).alias('CIG_CeilingHeightDim_stdv'), \
    ))

# COMMAND ----------

pd_joined = df_aggregated.toPandas()


# COMMAND ----------

figure, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(ax=axes[0], data = pd_joined, x = 'DEP_DEL15', y = 'DEW_Value', palette='Blues').set(title = "Flight departure delays and dew point temperature", xlabel = "Departure delay indicator", ylabel = "dew point temperature (celcius x10)")
sns.boxplot(ax=axes[1], data = pd_joined, x = 'DEP_DEL15', y = 'CIG_CeilingHeightDim', palette='Blues').set(title = "Flight departure delays and cloud height", xlabel = "Departure delay indicator", ylabel = "Height above lowest cloud (meters)")
sns.boxplot(ax=axes[2], data = pd_joined, x = 'DEP_DEL15', y = 'AJ1_SnowDepth', palette='Blues').set(title = "Flight departure delays and snow accumulation", xlabel = "Departure delay indicator", ylabel = "Snow depth (cms)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation between numerical weather features

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import seaborn as sns
import  numpy as np

numerical_weather_cols = ['TMP_Value', 'DEW_Value', 'VIS_Horizontal', 'SLP_Value', 'AA_RainDepth', 'AA_RainDuration', 'AL_SnowAccumDuration', 'AJ1_SnowDepth', 'AJ1_SnowEqWaterDepth', 'WND_Speed', 'WND_DirectionAngle', 'CIG_CeilingHeightDim']
outcome_col = ['DEP_DEL15']
input_cols = outcome_col + numerical_weather_cols

outcome = 'corr_features'
df_agg_clean = df_aggregated.select(*input_cols).fillna(0)
assembler = VectorAssembler(inputCols=input_cols, outputCol=outcome, handleInvalid='skip')
weather_vector = assembler.transform(df_agg_clean).select(outcome)
matrix = Correlation.corr(weather_vector, outcome,'pearson').collect()[0][0]
corrmatrix = np.round(matrix.toArray(),3).tolist()

# plot the heat map
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrmatrix, annot=True, fmt="g", cmap="YlGnBu", ax=ax)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checking out the top present weather conditions for delayed and non-delayed flights, before and after aggregation

# COMMAND ----------

# Before aggregating
num_delayed_flights = df_joined.filter(col('DEP_DEL15') == 1).count()
num_not_delayed_flights = df_joined.filter(col('DEP_DEL15') == 0).count()
display(df_joined.groupBy('DEP_DEL15', 'AW1_PresentWeatherCond').count().withColumn('pct', F.when(col('DEP_DEL15') == 1, col('count') / num_delayed_flights).otherwise(col('count') / num_not_delayed_flights)).orderBy('DEP_DEL15',F.desc('pct')))

# COMMAND ----------

num_delayed_flights_agg = df_aggregated.filter(col('DEP_DEL15') == 1).count()
num_not_delayed_flights_agg = df_aggregated.filter(col('DEP_DEL15') == 0).count()
df_weather_conditions_agg = df_aggregated.groupBy('DEP_DEL15', 'AW1_PresentWeatherCond').count().withColumn('pct', F.when(col('DEP_DEL15') == 1, col('count') / num_delayed_flights_agg).otherwise(col('count') / num_not_delayed_flights_agg)).orderBy('DEP_DEL15',F.desc('pct'))


# COMMAND ----------

display(df_weather_conditions_agg)

# COMMAND ----------

# MAGIC %md
# MAGIC Before aggregation, the top present weather conditions are:  
# MAGIC <b>Delayed flights</b>
# MAGIC   * 65% null  
# MAGIC   * 23.5% '10' = 'Mist'
# MAGIC   * 3.3% '71' = 'Snow, slight'  
# MAGIC   * 2.5% '61' = 'Rain, not freezing, slight'  
# MAGIC   * 1.5% '30' = 'Fog'  
# MAGIC <b>Non-delayed flights</b>
# MAGIC   * 80% null  
# MAGIC   * 13.5% '10' = 'Mist'
# MAGIC   * 2% '61' = 'Rain, not freezing, slight'
# MAGIC   * 1.5% '71' = 'Snow, slight'
# MAGIC   
# MAGIC   
# MAGIC After aggregation, the top present weather conditions are:  
# MAGIC <b>Delayed flights</b>
# MAGIC   * 64% null  
# MAGIC   * 23.2% '10' = 'Mist'
# MAGIC   * 4.6% '61' = 'Rain, not freezing, slight'   
# MAGIC   * 2.8% '71' = 'Snow, slight'  
# MAGIC   * 1.7% '30' = 'Fog'  
# MAGIC <b>Non-delayed flights</b>
# MAGIC * 77.7% null  
# MAGIC * 14.4% '10' = 'Mist'
# MAGIC * 4.2% '61' = 'Rain, not freezing, slight'
# MAGIC * 1.5% '71' = 'Snow, slight'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checking out categorical Wind Type indicator, before and after aggregation

# COMMAND ----------

# Before aggregation
display(df_joined.groupBy('DEP_DEL15', 'WND_Type').count().withColumn('pct', F.when(col('DEP_DEL15') == 1, col('count') / num_delayed_flights).otherwise(col('count') / num_not_delayed_flights) ).orderBy('DEP_DEL15', F.desc('pct')))

# COMMAND ----------

# After aggregation
display(df_aggregated.groupBy('DEP_DEL15', 'WND_Type').count().withColumn('pct', F.when(col('DEP_DEL15') == 1, col('count') / num_delayed_flights_agg).otherwise(col('count') / num_not_delayed_flights_agg) ).orderBy('DEP_DEL15', F.desc('pct')))

# COMMAND ----------

num_delayed_flights_agg = df_aggregated.filter(col('DEP_DEL15') == 1).count()
num_not_delayed_flights_agg = df_aggregated.filter(col('DEP_DEL15') == 0).count()
wind_type_delays = df_aggregated.groupBy('DEP_DEL15', 'WND_Type').count().withColumn('pct', F.when(col('DEP_DEL15') == 1, col('count') / num_delayed_flights_agg).otherwise(col('count') / num_not_delayed_flights_agg) ).orderBy('DEP_DEL15', F.desc('pct')).cache()
pd_wind_type_delays = wind_type_delays.drop('count').filter(col('WND_Type').isNotNull()).toPandas() 

# COMMAND ----------

wind_type_pivot = pd_wind_type_delays.pivot(index='DEP_DEL15', columns=['WND_Type'], values='pct')
colors = sns.color_palette("Blues", n_colors=4)
cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors)
wind_type_pivot.plot(kind='bar', stacked=True, colormap=cmap1)

# COMMAND ----------


window_condition = Window.partitionBy(['DEP_DEL15']).orderBy(F.desc('count'))
weather_cond_delays = df_aggregated.groupBy('DEP_DEL15', 'AW1_PresentWeatherCond').count().withColumn('pct', F.when(col('DEP_DEL15') == 1, col('count') / num_delayed_flights_agg).otherwise(col('count') / num_not_delayed_flights_agg)).withColumn("Rank", F.row_number().over(window_condition)).filter(col("Rank") <= 5)


# COMMAND ----------

pd_weather_cond_delays = weather_cond_delays.drop('count').filter(col('AW1_PresentWeatherCond').isNotNull()).toPandas() 

# COMMAND ----------

pd_weather_cond_delays

# COMMAND ----------

weather_cond_pivot = pd_weather_cond_delays.pivot(index='DEP_DEL15', columns=['AW1_PresentWeatherCond'], values='pct')
weather_cond_pivot.plot(kind='bar', stacked=True, colormap=cmap1)

# COMMAND ----------

colors = sns.color_palette("Blues", n_colors=4)
cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors)
weather_cond_pivot = pd_weather_cond_delays.pivot(index='DEP_DEL15', columns=['AW1_PresentWeatherCond'], values='pct')
weather_cond_pivot.plot(kind='bar', colormap=cmap1).set(xlabel='Delay indicator', ylabel='Occurance percentage', title='Top 4 weather conditions for delayed and non-delayed flights ')
leg = plt.legend()
leg.get_texts()[0].set_text('Mist')
leg.get_texts()[1].set_text('Smoke')
leg.get_texts()[2].set_text('Rain, not freezing, slight')
leg.get_texts()[3].set_text('Snow, slight')


# COMMAND ----------

