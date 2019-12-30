# sparkify
Data Science Capstone Project on predicting customer churn

## 1. Installations ##
import numpy as np<br/>
import pandas as pd<br/>
import seaborn as sns<br/>
from pyspark.sql import SparkSession, Window<br/>
import pyspark.sql.functions as F<br/>
from pyspark.sql.types import IntegerType, DateType, TimestampType, StringType, DoubleType, LongType<br/>
%matplotlib inline<br/>
import matplotlib.pyplot as plt<br/>
import datetime<br/>
from pyspark.ml.feature import VectorAssembler, MinMaxScaler<br/>
from pyspark.ml import Pipeline<br/>
from pyspark.ml.evaluation import BinaryClassificationEvaluator<br/>
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator<br/>
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier<br/>
from sklearn.metrics import f1_score, recall_score, precision_score<br/>
from pyspark.ml.feature import PCA<br/>
import time <br/>

## 2. Project Motivation ##
This project analyzes customer churn on the example of a fictive music streaming service called "Sparkify". The purpose is to show how such an analysis is done for large datasets by using Spark. The project is part of a Udacity Nanodegree project and thus for educational purposes.

## 3. File Descriptions ##
"Sparkify.ipynb": Jupyter Notebook used for analysis of Sparkify user log data on IBM Watson Studio Platform.
"Sparkify.md": mark-down version of the Jupyter Notebook for better readability.
"Sparkify_prep.ipynb": Jupyter Notebook used for analysis of Sparkify user log data in Udacity Classroom.
"Sparkify_prep.md": mark-down version of the Jupyter Notebook for better readability.

## 4. Results and Conclusion ##
Via this repository it was demonstrated how Spark can be applied to a binary classification task such as customer churn detection for large data sets.
Firstly a Spark session was started and the data set loaded. After some preliminary analysis “churn” was defined in the context of the available data. Data exploration was done to understand the available data. Since the analysis of churn relates to a specific user, a new Dataframe is created that summarizes the churn related data per user.<br/><br/>
Assumptions on possible user specific features that relate to churn are being made. These are then engineered from the original data set features and further analysed. Among these new features are some derived from personal user data (e.g. gender) and others derived from service usage (e.g. user page actions and usage time). The relationship of these features with regards to churn was graphically analyzed via pair plots and a correlation heatmap.<br/><br/>
The resulting 12 derived features are both binary and non-binary numeric features. The non-binary numeric features need to be scaled for further analysis. This was done via a MinMaxScaler.<br/><br/>
In order to reduce the feature space for modeling, Principal Component Analysis was applied. The features were reduced to three principal components which explain 88.20% of overall feature variance. This data is then split into a training and test set.<br/><br/>
A simple Logistic Regression Model showed poor performance (F1 Score: 0.30, Recall: 0.43, Precision: 0.23). Further model tuning of the “maxIter” hyperparameter increased model performance slightly (F1 Score: 0.37, Recall: 0.62, Precision: 0.27).<br/><br/>
A Decision Tree Model was as evaluated as alternative algorithm. A simple model without tuning yielded better results on F1 Score and Precision, but worse on Recall (F1 Score: 0.40, Recall: 0.29, Precision: 0.67). Tuning of this model via the hyperparameters “maxBins” and “impurity” resulted in perfect Precision but worsened F1 Score and Recall (F1 Score: 0.25, Recall: 0.14, Precision: 1.00).<br/><br/>
Consequently model selection appears to be a performance trade-off between Precision and Recall. This is dependent on the business cost of False Negatives and False Positives. If the cost of False Negatives is high, a Logistic Regression Model should be chosen. However if the cost of False Positives is high, a Decision Tree Model should be the choice.<br/><br/>
Possible model improvement could be made with different features, that provide better insights on Sparkify’s product and service perception. Maybe the existing data could provide such features. If not additional data would need to be collected. Also different estimators and further tuning of hyperparameters might improve model performance.

## 5. How to Interact with your project ##
Since the main purpose of this analysis is educational, I would really appreciate any feedback on possible improvements in code, modeling and general problem-solving approach. Otherwise it would be may pleasure if my files could serve as orientation or base for other projects.
