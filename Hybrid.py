#!/usr/bin/env python
# coding: utf-8



#Importing findspark to run the code on the local machine
#import findspark
#findspark.init()




#Importing required Libraries
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, DateType, FloatType, ArrayType
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.functions import col, unix_timestamp, to_date
from math import sin, cos, sqrt, atan2, radians, asin
from pyspark.sql import functions as F
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer
from pyspark.sql.functions import udf
from pyspark.ml.regression import DecisionTreeRegressor
import re, math
from pyspark.ml.clustering import LDA
import string
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, IDF
import matplotlib.pyplot as plt

from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col


import numpy as np

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
#from pyspark.mllib.recommendation import ALS, Rating





spark = SparkSession.builder.master("yarn").appName("Hybrid").getOrCreate()
#Create a spark context in session
sc = spark.sparkContext
#Reduce output by only showing me the errors
sc.setLogLevel("ERROR")
#SQL Context
sqlContext = SQLContext(sc)
sc.setCheckpointDir('checkpoint')





#Creating a schema for the ratings csv file
schema = StructType([
    StructField("UserId", IntegerType(), True),
    StructField("MovieId", IntegerType(), True),
    StructField("Rating", FloatType(), True)])





#Reading our ratings file with the pre-defined schema
ratings = spark.read.csv("ratings.csv",schema=schema,header=True)



(#Splitting our dataset into test and train
(train,test) = ratings.randomSplit([0.8,0.2])



#ALS aims to fill out the missing entries of a user-item association matrix described by a small set of latent factors.
#coldStartStrategy = ‘drop’ – It handles missing or unknown values by dropping entries that have not been seen by the model.
#nonnegative = ‘True’ - Removes negative values for prediction

alsModel = ALS(coldStartStrategy="drop",userCol='UserId',itemCol='MovieId',ratingCol='Rating', nonnegative=True)b





#The ALS model is cross validated on a pre-defined parameter grid which consists of:
#MaxIter = [20,50,100]
#regParam i.e. The Regularization parameter = [0.1,0.01,0.5]
#rank – It is an important parameter in ALS that deals with the number of features i.e. 
#the latent factors to use for prediction. = [10,50,100]


paramGrid = (ParamGridBuilder()\             
.addGrid(alsModel.maxIter, [20,50,100])\             
.addGrid(alsModel.regParam, [0.1,0.01,0.5])\
.addGrid(alsModel.rank,[10,50,100])\
.build())   

evaluator_rmse = RegressionEvaluator(predictionCol="prediction", labelCol="Rating", metricName="rmse")
evaluator_mae = RegressionEvaluator(predictionCol="prediction", labelCol="Rating", metricName="mae")

#Num_Folds are 3 due to the size of the data
cv=  CrossValidator(estimator=alsModel,estimatorParamMaps=paramGrid,evaluator=evaluator_rmse, numFolds=3)




model = cv.fit(train)

#Getting our best model from the cross validated models
final_model = model.bestModel


#Reading our created utility matrix from the moviePairSimilarities.py code
def read_similarity_mat(x):
    arr = x.replace(" ", "").split(",")
    return (int(arr[0][2:]), int(arr[1][:-1])), (float(arr[2][1:]), int(arr[3][:-2]))

similarity_mat = sc.textFile("movie-sims.txt").map(read_similarity_mat)


#Converting the similarity matrix into a dataframe with the predefined schema
sim_schema = StructType([
    StructField("mv_id1", IntegerType(), True),
    StructField("mv_id2", IntegerType(), True),
    StructField("sim", DoubleType(), True)])

sim_mat_df = sqlContext.createDataFrame(similarity_mat.map(lambda x: (x[0][0], x[0][1], x[1][0])), sim_schema)

test_df = test.withColumnRenamed("UserId","user_id").    withColumnRenamed("MovieId","test_mv_id").    withColumnRenamed("rating","target_rating")
train_df = train.withColumnRenamed("UserId","train_user_id").    withColumnRenamed("MovieId","train_mv_id").    withColumnRenamed("rating","train_rating")


# ## Steps to compute Item-Item CF
# 1. Join test set (`user_id`, `test_mv_id`, `rating`) and similarity matrix (`mv_id1`, `mv_id2`, `sim`) to get similarities attached to test movie ids `test_mv_id` with every other movie id `mv_id2`.
#     
#     test set **JOIN ON `test_mv_id` == `mv_id1`** similarity matrix $\longrightarrow$ (`user_id`, `test_mv_id`, `target_rating`,  `mv_id2`, `sim`)
# 2. Filter `mv_id2` to keep only movies rated by `user_id`. Join with train set (`train_user_id`, `train_mv_id`, `train_rating`)
# 
#     test set **JOIN ON `user_id` == `train_user_id` and `mv_id2`== `train_mv_id`** train set $\longrightarrow$ (`user_id`, `test_mv_id`, `target_rating`,  `mv_id2`, `sim`, `mv2_rating`) (Exclude attributes from training set)
# 3. 
#     * Group by (`test_mv_id`, `user_id`), 
#     * sort desc (`sim`), 
#     * limit **N** rows in each group, 
#     * **sum(`sim`, `sim`$\times$`mv2_rating`)** as `sum_sim`, `sum_sim_rating`
#     * `pred` = `sum_sim_rating`/`sum_sim`
#     (`user_id`, `test_mv_id`, `rating`,  `mv_id2`, `sim`, `mv2_rating`, `sum_sim`, `sum_sim_rating`, `pred`)
# 4. RMSE(`rating`, `pred`)




# Step 1
res1 = test_df.join(sim_mat_df, test_df.test_mv_id == sim_mat_df.mv_id1, how="inner").drop("mv_id1")





# Step 2
res2 = res1.join(train_df, (res1.user_id == train_df.train_user_id) & (res1.mv_id2 == train_df.train_mv_id), how="inner")        .withColumnRenamed("train_rating","mv_id2_rating")        .drop('train_user_id', 'train_mv_id')
res2 = res2.withColumn("sim*mv2rating", res2.sim * res2.mv_id2_rating)




# Step 3
window = Window    .partitionBy(res2['user_id'], res2["test_mv_id"])     .orderBy(res2['sim'].desc()) 

res3 = res2.select('*', rank().over(window).alias('rank'))    .filter(col('rank') <= 10)





res3_temp = res3.groupBy("user_id", "test_mv_id").sum('sim', 'sim*mv2rating')





preds = res3_temp.withColumn('prediction', res3_temp['sum(sim*mv2rating)']/res3_temp['sum(sim)']).persist()




iicf = test_df.select(col("user_id").alias("test_df_user_id"), 
                      col("test_mv_id").alias("test_df_test_mv_id"), 
                      col('target_rating').alias('Rating'))\
        .join(preds, (col("test_df_user_id") == col('user_id')) & (col('test_df_test_mv_id') == col('test_mv_id')), how="inner")\
        .drop('test_df_user_id', 'test_df_test_mv_id', 'sum(sim)', 'sum(sim*mv2rating)')




als_iicf_comb = predictions.join(iicf.select(col('user_id'), col('test_mv_id'), col('prediction').alias('iicf_pred')),
                (col('UserId') == col('user_id'))&(col('MovieId') == col('test_mv_id')), how="inner")\
                .drop('user_id', 'test_mv_id')\
                .withColumnRenamed("prediction", "als_pred")\
                .persist()




als_weight = 0.7
final_df = als_iicf_comb.withColumn('prediction', als_weight*col("als_pred") + (1-als_weight)*col('iicf_pred'))




rmse = evaluator_rmse.evaluate(final_df)
mae = evaluator_mae.evaluate(final_df)



print("RMSE of the results: ",rmse)
print("MAE of the results: ",mae)

