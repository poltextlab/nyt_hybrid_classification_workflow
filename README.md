# README
# Rebuilding the database

There are two ways to replicate the results of the paper. One can re-build the database by using the NYT API. For this way, use the "preproc_in_r/00_nyt_articles_api.R", in combination with the "data/valid_start.csv". The input of valid_start.csv is the file containing the dates and titles from the Boydstun (2013) dataset. 

The script contains a function to access the NYTimes API and download the lead paragraphs and other article metadata used in the article. In order to replicate the code, a working API key should be added at line 58.

The NYTimes API also imposes two rate limits: 4000 requests per day and 10 requests per minute. This means that accessing all 28 548 lead paragraphs takes around 8 days.

The "preproc_in_r/00_nyt_articles_api.R" script contains extensive exploration of the downloaded data (there are a few possible API errors that might need troubleshooting). These are optional to run, but they are included for completeness sake. As a result it produces the downloaded data file ("nyt_api_corpus.csv") that is the input of the 01_NYT_SPARK_PREPROC.R. 

Using the 01_NYT_SPARK_PREPROC.R for finalizing the downloaded data. This R script merges the lead paragraphs from the NTYtimes API, with the Boydstun (2013) dataset. This results in the starting corpus we use that contains the lead paragraphs, the necessary metadata and CAP coding.

This script applies the preprocessing steps described in the article: stemming, removing numbers, punctuation marks, white space, transforming to lowercase and removing stopwords.



--------------------------------------------------------------------------------

# Using the replication data provided in the repository

If the database rebuilding is not an option, you can use the cleaned "Data_NYT_clean_SPARK_START_sim.csv" as input for the analysis.

## Specifications for the Spark cluster

The rest of the steps (with the exception of the baseline models run for comparison) run in an Apache Spark cluster. The code has been tested to run with the following cluster setup and software environment:
1) 1 Master and 4 Executor nodes
2) All 5 VMs have 32GBs of RAM and 8 VCPUs
3) Ubuntu 16.04.6 LTS
4) Hadoop 2.9.2
5) Spark 2.4.0
6) Spark is run in standalone mode (https://spark.apache.org/docs/latest/spark-standalone.html)
7) the Python environment is:
	a) python 3.7.2
	b) pyspark 2.4.0
	c) numpy 1.16.2
	d) pandas 0.24.1

Please note that all instances of the IP address of the Spark Master in the code (currently 192.168.0.182) have to be changed to the actual IP address of the Spark Master in the cluster being used.


## Copying files to the Spark Master

Please copy the following files onto the Spark Master:
	1) "Data_NYT_clean_SPARK_START_sim.csv" (either the one created with the R script, or the one already provided in the data/ folder)
	2) Python scripts from the folders preproc_in_Spark/ and spark_cluster/.


## Setting up the HDFS for working with the code

The code assumes that an /input and an /eventlog directory have been created in the HDFS root directory.
The commands to create these directories are:
    "/usr/local/hadoop/bin/hadoop fs -mkdir /input"
    "/usr/local/hadoop/bin/hadoop fs -mkdir /eventlog"
Please note that if Hadoop is installed under a different path, then the path has to be adjusted accordingly in the above commands.

The starting table has to be loaded into the HDFS to the /input directory. First change to the directory where the Data_NYT_clean_SPARK_START_sim.csv file is located on the Spark Master, then issue the following command:
    "/usr/local/hadoop/bin/hadoop fs -put Data_NYT_clean_SPARK_START_sim.csv /input"
Please note that if Hadoop is installed under a different path, then the path has to be adjusted accordingly in the above command.


## Preprocessing in Spark


Please note that, if the above specified python environment is found under a virtual environment, then that virtual environment has to be activated first (if it is not the active environment by default) in order for the packages of the environment to be available for the code.

To run the Spark preprocessing script change to the folder where NYT_prepare_corpus.py is located on the Spark Master, then issue the following command:
"python3 NYT_prepare_corpus.py"



## Classification rounds

The scripts for the reproduction of each classification can be found in the spark_cluster/ folder. The subfolders are named in line with the terminology used in the article.

The python scripts for each round of classification for each approach should be run consecutively after the previous scripts finished. For convenience there are bash scripts that run each step one after another for folders that contain multiple scripts (see folder names beginning with 04_).

## Baseline models

The Random Forest, SVM and LASSO regression baseline models are done in the "baseline_models/02_lr_rf.R" script, using the "data/Data_NYT_clean_SPARK_START_sim.csv" file.

## Figures

Figures can be reproduced by using the "figures.R" script and the "data/output" folder. The data/output folder contains the output of our SPARK cluster results.

