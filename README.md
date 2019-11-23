# sparkify
Data Science Capstone Project on predicting customer churn

1. Installations
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, DateType, TimestampType, StringType, DoubleType, LongType
%matplotlib inline
import matplotlib.pyplot as plt
import datetime
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from pyspark.ml.feature import PCA


2. Project Motivation
This project analyzes customer churn on the example of a fictive music streaming service called "Sparkify". The purpose is to show how such an analysis is done for large datasets by using Spark. The project is part of a Udacity Nanodegree project and thus for educational purposes.

3. File Descriptions
"Sparkify.ipynb": Jupyter Notebook used for analysis of Sparkify user log data.
"Sparkify.md!: mark-down version of the Jupyter Notebook for better readability.

4. How to Interact with your project
Since the main purpose of this analysis is educational, I would really appreciate any feedback on possible improvements in code, modeling and general problem-solving approach. Otherwise it would be may pleasure if my files could serve as orientation or base for other projects. The projects results are the following:
Prediction of customer churn on the user log data can be made an accuracy of XX.
The three principal feature components explaince XX of variance in the user data.
