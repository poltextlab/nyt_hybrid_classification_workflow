# import libraries
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *

from pyspark.sql.functions import col, count, when

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
spark = SparkSession.builder.appName('ML1_wf_on_NYT_with_NB_sim2_and_sim3_to_sim1_round1').config(conf=conf).getOrCreate()

sc = spark.sparkContext

# check things are working
print(sc)
print(sc.defaultParallelism)
print("SPARK CONTEXT IS RUNNING")


#################################################
# define major topic codes
#################################################

# major topic codes for loop (NO 23 IN THE NYT CORPUS)
majortopic_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 100]
#majortopic_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 100]


#################################################
# loop starts here
#################################################

for h in range(7):
    # read table from hdfs
    df_original = spark.read.parquet("hdfs://192.168.0.182:9000/input/Data_NYT_clean_SPARK_START_ML2_features.parquet").repartition(50)

    # check loaded data 
    print(df_original.printSchema())
    print(df_original.show())
    df_original.groupBy("majortopic").count().show(30, False)

    #################################################
    # prepare to log sample numbers
    #################################################

    columns = ["label", "non_label_all", "non_label_sample", "train_all"]

    df_numbers = pd.DataFrame(index=majortopic_codes, columns=columns)

    for i in majortopic_codes:
        #################################################
        # prepare df for svm requirements
        #################################################
        print("majortopic is:", i)

        # separate majortopic
        df_original = df_original.withColumn("label", when(df_original["majortopic"] == i, 1).otherwise(0))

        # label has to be double for SVM
        df_original = df_original.withColumn('label', df_original.label.cast(DoubleType()))

        #################################################
        # separate training and test sets
        #################################################

        df_train = df_original.where(col('sim') != 1)
        df_test = df_original.where(col('sim') == 1)

        # make training data proportional with regards to label occurrence frequency
        df_train_mtc = df_train.where(col('label') == 1)
        df_train_non_mtc = df_train.where(col('label') == 0)

        df_train_count = df_train.count()
        df_train_mtc_count = df_train_mtc.count()
        df_train_non_mtc_count = df_train_non_mtc.count()
        print("Rows in training DataFrame with label = ", df_train_mtc_count)
        print("Rows in training DataFrame without label = ", df_train_non_mtc_count)

        if df_train_mtc_count/df_train_non_mtc_count < 0.1:
            if df_train_mtc_count*10 < df_train_count//10:
                sample_num = df_train_count//10
            else: sample_num = df_train_mtc_count*10
            print("sample_num = ", sample_num)
            print("df_train_non_mtc = ", df_train_non_mtc_count)
            sampling_fraction = sample_num/df_train_non_mtc_count
            print("sampling_fraction = ", sampling_fraction)
            df_train_non_mtc = df_train_non_mtc.sample(False, sampling_fraction)
            df_train_non_mtc_sample = df_train_non_mtc.count()
            print("Rows in training DataFrame without label = ", df_train_non_mtc_sample)
            df_train = df_train_mtc.union(df_train_non_mtc)
            # numbers to logtable
            df_numbers["non_label_sample"].loc[i] = df_train_non_mtc_sample
            df_numbers["train_all"].loc[i] = df_train_mtc_count + df_train_non_mtc_sample
        else:
            # numbers to logtable
            df_numbers["non_label_sample"].loc[i] = df_train_non_mtc_count
            df_numbers["train_all"].loc[i] = df_train_count

        # numbers to logtable
        df_numbers["label"].loc[i] = df_train_mtc_count
        df_numbers["non_label_all"].loc[i] = df_train_non_mtc_count
        print(df_numbers)

        # NOTE: this type of copying wouldn't work in python, but does work in pyspark!
        df_train_orig = df_train
        df_test_orig = df_test
        df_loop = 0
        df_train_mtc = 0
        df_train_non_mtc = 0

        print("Rows in training DataFrame = ", df_train.count())
        print("Rows in test DataFrame = ", df_test.count())


        #################################################
        # naive bayes
        #################################################

        for j in range(7):
            df_train = df_train_orig
            df_test = df_test_orig

            nb = NaiveBayes(featuresCol='features', labelCol='label', predictionCol='prediction',
                            modelType='multinomial', smoothing=0.1)

            # train the model.
            nbModel = nb.fit(df_train)

            print("fit model finished, starting scoring:", j)

            # score the model on test data.
            predictions = nbModel.transform(df_test)

            df_train = 0
            df_test = 0
            nbModel = 0

            print(predictions.printSchema())
            print(predictions.show())

            df_write = predictions.select("doc_id", "prediction")

            predictions = 0

            df_write = df_write.withColumn('prediction', df_write.prediction.cast(IntegerType()))
            df_write = df_write.withColumn('prediction', df_write.prediction * i)
            new_col_name = 'prediction_{i}'.format(i=i)
            df_write = df_write.withColumnRenamed('prediction', new_col_name)

            # write partial result to parquet
            dest_name = "hdfs://192.168.0.182:9000/input/NYT_prediction_mtc{i}_{j}.parquet".format(i=i, j=j)
            df_write.write.parquet(dest_name, mode="overwrite")

            df_write = 0

        print("DONE")

    print("ALL NB DONE round1_{h}".format(h=h+1))

    df_numbers.to_csv("NYT_round1_sample{h}_sample_numbers.csv".format(h=h+1), index=False)

    # empty memory
    spark.catalog.clearCache()
    print("cache cleared")

    #######################################################
    ### parquet to pandas
    #######################################################

    for j in range(7):
        # read from parquet format
        for i in majortopic_codes:
            source_name = "hdfs://192.168.0.182:9000/input/NYT_prediction_mtc{i}_{j}.parquet".format(i=i, j=j)
            df = spark.read.parquet(source_name).repartition(50)
            if i == 1:
                df_results = df
            else:
                df_results = df_results.join(df, 'doc_id', 'inner')

        df = df_results
        df_results = 0

        # convert prediction results to pandas df
        df = df.toPandas()

        df.to_csv("NYT_round1_sample{h}_nb{j}.csv".format(h=h+1,j=j), index=False)


#########################################################################
# create table of single verdicts
#########################################################################

# all of the following happen in pandas outside the spark context
for i in range(7):
    for j in range(7):
        df = pd.read_csv("NYT_round1_sample{i}_nb{j}.csv".format(i=i+1, j=j))
        df = df.sort_values(by=['doc_id'])
        df = df.reset_index(drop=True)
        #print(df.head())
        if i == 0 and j == 0:
            df_results = df
        else:
            df_lemma = df_results.iloc[:,1:].add(df.iloc[:,1:])
            df_results = pd.concat([df_results[['doc_id']], df_lemma], axis=1)
            #print(df_results.head())

for i in majortopic_codes:
    df_results[["prediction_{i}".format(i=i)]] = df_results[["prediction_{i}".format(i=i)]].floordiv(i)

df_results["max_value"] = df_results.iloc[:,1:].max(axis = 1, numeric_only = True)
df_results["how_many_49votes"] = df_results.iloc[:,:-1].isin([49]).sum(1)

# keep only rows with verdicts
print(df_results.shape)
df_results = df_results.loc[df_results["max_value"]==49]
print(df_results.shape)
# keep only rows with a single verdict
df_results = df_results.loc[df_results["how_many_49votes"]==1]
print(df_results.shape)

# prepare table of single verdicts
df_results = df_results.drop(['max_value', 'how_many_49votes'], axis=1)

print(df_results.head())

for i in majortopic_codes:
    df_results[["prediction_{i}".format(i=i)]] = df_results[["prediction_{i}".format(i=i)]].floordiv(49)

print(df_results.head())

for i in majortopic_codes:
    df_results[["prediction_{i}".format(i=i)]] = df_results[["prediction_{i}".format(i=i)]]*i


df_results["verdict_r1"] = df_results.iloc[:,1:].sum(1)

df_results = df_results[["doc_id", "verdict_r1"]]

# write to csv
df_results.to_csv("NYT_round1_single_verdicts.csv", index=False)

# to write results to parquet for use in the next round we need to move the pandas df into a spark df
df_results = spark.createDataFrame(df_results)

df_results.write.parquet("hdfs://192.168.0.182:9000/input/NYT_round1_single_verdicts_NB.parquet", mode="overwrite")

sc.stop()
spark.stop()
