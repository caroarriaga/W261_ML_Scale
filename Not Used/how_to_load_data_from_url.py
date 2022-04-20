# Databricks notebook source
url = "https://pkgstore.datahub.io/machine-learning/iris/iris_csv/data/8bce8766530bf404228ea3fc026dfee3/iris_csv.csv"
from pyspark import SparkFiles
spark.sparkContext.addFile(url)

# COMMAND ----------

import os
filename = SparkFiles.get(url.split("/")[-1])
env_var = filename.split('.')[0]
os.environ[env_var] = filename

# COMMAND ----------

# MAGIC %%bash
# MAGIC mv $url_iris /dbfs/
# MAGIC ls /dbfs

# COMMAND ----------

df = spark.read.csv("/iris_csv.csv", header=True, inferSchema= True)
df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC After this follow the same steps to write this data to your Azure Blob Storage as we have showed in the `mount_team_cloud_storage` notebook.

# COMMAND ----------

