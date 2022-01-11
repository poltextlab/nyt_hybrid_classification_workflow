# import libraries
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *

from pyspark.sql.functions import col, count, when

from pyspark.ml.classification import LinearSVC

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
spark = SparkSession.builder.appName('ML2_HV_v4_activeLearn_NYT_sim2_and_sim3_to_sim1_round2').config(conf=conf).getOrCreate()

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

for h in range(3):
    # read table from hdfs
    df_original = spark.read.parquet("hdfs://192.168.0.182:9000/input/ML2_HV_v4_activeLearn_NYT_round2_start.parquet").repartition(50)

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

        df_train = df_original.where((col('train_r2') == 1) | (col('train_r2_neg') == i))
        df_test = df_original.where((col('train_r2') == 0) & (col('train_r2_neg') != i))

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
        # SVM
        #################################################

        for j in range(3):
            df_train = df_train_orig
            df_test = df_test_orig

            # define svm
            lsvc = LinearSVC(featuresCol='features', labelCol='label', maxIter=10, regParam=0.1)

            # train the model.
            lsvcModel = lsvc.fit(df_train)

            print("fit model finished, starting scoring:", j)

            # score the model on test data.
            predictions = lsvcModel.transform(df_test)

            df_train = 0
            df_test = 0
            lsvcModel = 0

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

    print("ALL SVM DONE round2_{h}".format(h=h+1))

    df_numbers.to_csv("ML2_HV_v4_activeLearn_NYT_round2_sample{h}_sample_numbers.csv".format(h=h+1), index=False)

    # empty memory
    spark.catalog.clearCache()
    print("cache cleared")

    #######################################################
    ### parquet to pandas
    #######################################################

    for j in range(3):
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

        df.to_csv("ML2_HV_v4_activeLearn_NYT_round2_sample{h}_svm{j}.csv".format(h=h+1,j=j), index=False)


#########################################################################
# create results and leftovers tables
#########################################################################

# all of the following happen in pandas outside the spark context
for i in range(3):
    for j in range(3):
        df = pd.read_csv("ML2_HV_v4_activeLearn_NYT_round2_sample{i}_svm{j}.csv".format(i=i+1, j=j))
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
df_results["how_many_9votes"] = df_results.iloc[:,:-1].isin([9]).sum(1)

print(df_results.shape)
df_results = df_results.loc[df_results["max_value"]==9]
print(df_results.shape)
# first get table of multiple nine votes for active learning
df_activeLearn = df_results.loc[df_results["how_many_9votes"]>1]
# then get all simple verdicts
df_results = df_results.loc[df_results["how_many_9votes"]==1]
print(df_results.shape)

# prepare table for active learning
# first get the full result table for further analysis later
df_activeLearn.to_csv("ML2_v4_activeLearn_NYT_r2_activeLearn_raw.csv", index=False)

# since this is a simulation a dummy value will suffice here
df_activeLearn["verdict"] = "dummy_value"
df_activeLearn = df_activeLearn[["doc_id", "verdict"]]

# prepare table of single verdicts
df_results = df_results.drop(['max_value', 'how_many_9votes'], axis=1)

print(df_results.head())

for i in majortopic_codes:
    df_results[["prediction_{i}".format(i=i)]] = df_results[["prediction_{i}".format(i=i)]].floordiv(9)

print(df_results.head())

for i in majortopic_codes:
    df_results[["prediction_{i}".format(i=i)]] = df_results[["prediction_{i}".format(i=i)]]*i


df_results["verdict"] = df_results.iloc[:,1:].sum(1)

df_results = df_results[["doc_id", "verdict"]]

# now we move back to the spark context!!
# for that we need to move the pandas df into a spark df
df = spark.createDataFrame(df_results)
df_al = spark.createDataFrame(df_activeLearn)

# load df_original
df_original = spark.read.parquet("hdfs://192.168.0.182:9000/input/ML2_HV_v4_activeLearn_NYT_round2_start.parquet").repartition(50)

# create results table
df_results = df_original.join(df, "doc_id", "inner")
df_results_al = df_original.join(df_al, "doc_id", "inner")

# create table of non-classified and training elements
ids_drop = df.select("doc_id")
df_original = df_original.join(ids_drop, "doc_id", "left_anti")
# once more for those selected for active learning
ids_drop = df_al.select("doc_id")
df_original = df_original.join(ids_drop, "doc_id", "left_anti")

# write to parquet for use in human validation script
df_original.write.parquet("hdfs://192.168.0.182:9000/input/ML2_HV_v4_activeLearn_NYT_r2_train_and_remaining_NOTclassified.parquet", mode="overwrite")
df_results.write.parquet("hdfs://192.168.0.182:9000/input/ML2_HV_v4_activeLearn_NYT_r2_classified.parquet", mode="overwrite")
df_results_al.write.parquet("hdfs://192.168.0.182:9000/input/ML2_HV_v4_activeLearn_NYT_r2_activeLearn.parquet", mode="overwrite")

# convert tables to pandas df and write to csv
df_original = df_original.drop("text", "words", "raw_features", "features").toPandas()
df_results = df_results.drop("text", "words", "raw_features", "features").toPandas()
df_results_al = df_results_al.drop("text", "words", "raw_features", "features").toPandas()

df_original.to_csv("ML2_HV_v4_activeLearn_NYT_r2_train_and_remaining_NOTclassified.csv", index=False)
df_results.to_csv("ML2_HV_v4_activeLearn_NYT_r2_classified.csv", index=False)
df_results_al.to_csv("ML2_HV_v4_activeLearn_NYT_r2_activeLearn.csv", index=False)

print("df_original: ", df_original.shape[0])
print("df_results: ", df_results.shape[0])
print("df_results_activeLearn: ", df_results_al.shape[0])

sc.stop()
spark.stop()
