#!/usr/bin/env python
# coding: utf-8



#Importing findspark to run the code on the local machine
#import findspark
#findspark.init()




#Importing required Libraries
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, DateType, FloatType, ArrayType
from pyspark.sql.functions import isnan, when, count, col

from pyspark.sql import functions as F

from pyspark.sql.functions import udf

import re

import string
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, IDF
import matplotlib.pyplot as plt
import numpy as np

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS




spark = SparkSession.builder.master("yarn").appName("ALS").getOrCreate()
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



#Splitting our dataset into test and train
(train,test) = ratings.randomSplit([0.8,0.2])



#ALS aims to fill out the missing entries of a user-item association matrix described by a small set of latent factors.
#coldStartStrategy = ‘drop’ – It handles missing or unknown values by dropping entries that have not been seen by the model.
#nonnegative = ‘True’ - Removes negative values for prediction

alsModel = ALS(coldStartStrategy="drop",userCol='UserId',itemCol='MovieId',ratingCol='Rating', nonnegative=True)




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
predictions = final_model.transform(test)
final_rmse = evaluator_rmse.evaluate(predictions)
final_mae = evaluator_mae.evaluate(predictions)


#Printing the RMSE, MAE
print("RMSE of the results: ",final_rmse)
print("MAE of the results: ",final_mae)


#Converting the test dataframe to a Rdd
ratings_rdd = test.rdd.map(list).map(lambda x: (x[0], x[1], x[2]))


#Taking user recommendations and converting them into an rdd of (userid,(movieid,rating))
user_recs = final_model.recommendForAllUsers(100)
user_recs = user_recs.rdd.map(list).flatMapValues(list).map(lambda x: (x[0], (x[1][0],x[1][1])))

#Filtering out the movies with high rating and grouping the rdd by every userId
user_recs = user_recs.filter(lambda x: x[1][1] > 2.0).map(lambda x:(x[0],x[1][0])).groupByKey().mapValues(list)


#Filtering out the actual good movies as rated by the user
good_movies = ratings_rdd.filter(lambda x: x[2]>2.0)
good_movies = good_movies.map(lambda x: (x[0],x[1])).groupByKey().mapValues(list)


#Joining the two rdds by the userID and getting an rdd of the format 
#[([1.0, 6.0, 2.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0]), ([4.0, 1.0, 9.0, 10.0], [1.0, 2.0, 3.0])]
predictionAndLabels = user_recs.join(good_movies)
test_predictionAndLabels = predictionAndLabels.map(lambda x: x[1])

#Calculating the meanAveragePrecision
metrics = RankingMetrics(test_predictionAndLabels)
meanAveragePrecision = metrics.meanAveragePrecision
print("mAP of the results: ",meanAveragePrecision)

