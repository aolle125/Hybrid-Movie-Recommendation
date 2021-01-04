# Hybrid-Movie-Recommendation

Files:

1) ALS.py & ALS.ipynb - Collaborative Filtering using Alternative Least Squares 
			matrix factorization technique 

2) Hybrid.py & Hybrid.ipynb - Hybrid System that uses the ALS solution and item-item CF

3) moviePairSimilarities.py - The code that calculates the similarity matrix between items i.e. movies


Python Files:

Run the spark-submit moviePairSimilarities.py

The other codes can be run using spark-submit <filename.py>

Jupyter Notebooks:

The code can be run in a jupyter notebook using the findspark package in Python

pip install findspark
import findspark
findspark.init()

This helps us find the local spark instance in our computer and makes our jupyter notebook compatible
with Pyspark

Configuration:

The following configuration settings could be used which worked best for us:
"spark.executor.memory": "14G",
"spark.executor.cores": "3", 
"spark.executor.instances": "5", 
"spark.driver.memory": "30G", 
"spark.driver.cores": "3", 
"spark.default.parallelism": "30" 
"livy.server.session.timeout":"8h" 


On Amazon EMR:

We made use of the EMR Notebooks in Amazon AWS EMR to perform our operations.
Our csv file was saved on a bucket on Amazon S3.
The same notebook could be run on EMR without the findspark initialization



