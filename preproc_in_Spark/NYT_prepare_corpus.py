# import libraries
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *

from pyspark.sql.functions import count, when

from pyspark.ml.feature import RegexTokenizer, CountVectorizer, IDF

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
spark = SparkSession.builder.appName('NYT_preproc').config(conf=conf).getOrCreate()

sc = spark.sparkContext

# check things are working
print(sc)
print(sc.defaultParallelism)
print("SPARK CONTEXT IS RUNNING")

#################################################
# read data from csv
#################################################
NYT_DF = spark.read.option("delimiter", ";").csv("hdfs://192.168.0.182:9000/input/Data_NYT_clean_SPARK_START_sim.csv",
                               inferSchema=False, header=True)

print("HDFS READ DONE")
print((NYT_DF.count(), len(NYT_DF.columns)))
print(NYT_DF.show(5, truncate=False))

# check columns
print(NYT_DF.printSchema())


#######################################################
### IO operations for spark specific data format
#######################################################

# NOTE: it is important to write the data to parquet format and then load it again in that form to achieve a scalable pipeline

# write to spark specific data format
NYT_DF.write.parquet("hdfs://192.168.0.182:9000/input/Data_NYT_clean_SPARK_START_ML2.parquet", mode="overwrite")
print("WRITE TO PARQUET COMPLETE")
NYT_DF=0

# read from parquet format
df = spark.read.parquet("hdfs://192.168.0.182:9000/input/Data_NYT_clean_SPARK_START_ML2.parquet").repartition(50)

# check loaded data 
print("df.is_cached = ", df.is_cached)
print("rows in df = ", df.count())
print("number of partition = ", df.rdd.getNumPartitions())


#################################################
# create bag of words representation
#################################################

# first we need to drop empty texts as they will raise an exception later on
df = df.na.drop()
print("rows in df = ", df.count())

# second we need to create lists from sentences!
tokenizer = RegexTokenizer(inputCol="text", minTokenLength=2, outputCol="words")
df = tokenizer.transform(df)

print(df.printSchema())
print(df.show())
print(df.select("words").show())
print("TOKENIZE IS DONE")

cv = CountVectorizer(inputCol="words", outputCol="raw_features", minDF=5.0)

model = cv.fit(df)

print("BOW MODEL DONE")

# apply IDF model to df
df = model.transform(df)

print(df.printSchema())
print("BOW TRANSFORM DONE")

idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(df)
df = idfModel.transform(df)

print(df.printSchema())
print("IDF TRANSFORM DONE")

# majortopic to integer
df = df.withColumn('majortopic', df.majortopic.cast(IntegerType()))

# recode media codes (>23) to other (100)
df = df.withColumn("majortopic", when(df["majortopic"] > 23, 100).otherwise(df["majortopic"]))

df.groupBy("majortopic").count().show(30, False)

# write to spark specific data format
df.write.parquet("hdfs://192.168.0.182:9000/input/Data_NYT_clean_SPARK_START_ML2_features.parquet", mode="overwrite")
print("WRITE TO PARQUET COMPLETE")

sc.stop()
spark.stop()

