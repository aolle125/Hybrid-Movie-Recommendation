#!/usr/bin/env python
# coding: utf-8

#import findspark
#findspark.init()




import sys
from pyspark import SparkConf, SparkContext
from math import sqrt
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.mllib.evaluation import RegressionMetrics
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
import re
from pyspark.ml.clustering import LDA
import string
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, IDF
import matplotlib.pyplot as plt
import numpy as np

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS




spark = SparkSession.builder.master("local").appName("Sim").getOrCreate()
#Create a spark context in session
sc = spark.sparkContext
#Reduce output by only showing me the errors
sc.setLogLevel("ERROR")
#SQL Context
sqlContext = SQLContext(sc)
sc.setCheckpointDir('checkpoint')



def loadMovieNames():
    movieNames = {}
    with open("movies.csv", encoding='ascii', errors="ignore") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames



nameDict = loadMovieNames()


#Creating a schema for the ratings csv file
schema = StructType([
    StructField("UserId", IntegerType(), True),
    StructField("MovieId", IntegerType(), True),
    StructField("Rating", FloatType(), True)])


#Reading our ratings file with the pre-defined schema
ratings = spark.read.csv("ratings.csv",schema=schema,header=True)


#Converting our ratings to an rdd
ratings_rdd = ratings.rdd.map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))


#Giving 1000 partitions to increase parallelism
ratingsPartitioned = ratings_rdd.partitionBy(1000)
# Self-join to find every combination
joinedRatings = ratingsPartitioned.join(ratingsPartitioned)



#Making pairs of movie and ratings
def makePairs(user_ratings):
    _, ratings = user_ratings
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))



#Removing duplicate movie pairs from our set
def filterDuplicates( user_ratings ):
    _, ratings = user_ratings
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2



#Computing the cosine similarity by the formula = (A*B)/(||A||*||B||)
def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1
    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)
    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))
    return (score, numPairs)



#Removing duplicate movie pairs 
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)



# Now key by (movie1, movie2) pairs.
moviePairs = uniqueJoinedRatings.map(makePairs).partitionBy(100)

# We now have (movie1, movie2) => (rating1, rating2)
# Now collect all ratings for each movie pair and compute similarity

moviePairRatings = moviePairs.groupByKey()



#Computing the moviePairSimilarities
moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).persist()

moviePairSimilarities.saveAsTextFile("movie-sims.txt")

