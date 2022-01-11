# import libraries
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *

from pyspark.sql.functions import col, count

from pyspark.ml.classification import NaiveBayes

import pandas as pd

#################################################
# spark config
#################################################
mtaMaster = "spark://192.168.0.182:7077"

conf = SparkConf()
conf.setMaster(mtaMaster)

conf.set("spark.executor.memory", "24g")
conf.set("spark.driver.memory", "26g")
conf.set("spark.cores.max", 96)
conf.set("spark.driver.cores", 8)

conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.kryoserializer.buffer", "256m")
conf.set("spark.kryoserializer.buffer.max", "256m")

conf.set("spark.default.parallelism", 24)

conf.set("spark.eventLog.enabled", "true")
conf.set("spark.eventLog.dir", "hdfs://192.168.0.182:9000/eventlog")
conf.set("spark.history.fs.logDirectory", "hdfs://192.168.0.182:9000/eventlog")

conf.set("spark.driver.maxResultSize", "2g")

conf.getAll()

#################################################
# create spark session
#################################################
spark = SparkSession.builder.appName('ML2_NYT_simple_NB_sim2_and_sim3_to_sim1').config(conf=conf).getOrCreate()

sc = spark.sparkContext

# check things are working
print(sc)
print(sc.defaultParallelism)
print("SPARK CONTEXT IS RUNNING")


#######################################################
# load table
#######################################################

# read from parquet format
df = spark.read.parquet("hdfs://192.168.0.182:9000/input/Data_NYT_clean_SPARK_START_ML2_features.parquet").repartition(50)

#################################################
# separate training and test sets
#################################################

df_train = df.where(col('sim') != 1)
df_test = df.where(col('sim') == 1)


#################################################
# naive bayes
#################################################

nb = NaiveBayes(featuresCol='features', labelCol='majortopic', predictionCol='nbPrediction',
                smoothing=0.1, modelType='multinomial')

nbModel = nb.fit(df_train)

nbPred = nbModel.transform(df_test)

df = nbPred.drop("text", "words", "raw_features", "features").toPandas()

df.to_csv("ML2_NYT_simple_NB_sim2_and_sim3_to_sim1.csv", index=False)

nbPred.show()

sc.stop()
spark.stop()
