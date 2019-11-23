
# Sparkify Project Workspace
This workspace contains a tiny subset (128MB) of the full dataset available (12GB). Feel free to use this workspace to build your project, or to explore a smaller subset with Spark before deploying your cluster on the cloud. Instructions for setting up your Spark cluster is included in the last lesson of the Extracurricular Spark Course content.

You can follow the steps below to guide your data analysis and model building portion of this project.


```python
# import libraries

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

# for model evaluation
from sklearn.metrics import f1_score, recall_score, precision_score

# for Principal Component Analysis
from pyspark.ml.feature import PCA

```


```python
# create a Spark session
spark = SparkSession \
    .builder \
    .appName("Sparkify") \
    .getOrCreate()
```


```python
# Install additional libraries via pip in the current Jupyter kernel
import sys
!{sys.executable} -m pip install pixiedust
```

    Requirement already satisfied: pixiedust in /opt/conda/lib/python3.6/site-packages (1.1.17)
    Requirement already satisfied: markdown in /opt/conda/lib/python3.6/site-packages (from pixiedust) (2.6.9)
    Requirement already satisfied: colour in /opt/conda/lib/python3.6/site-packages (from pixiedust) (0.1.5)
    Requirement already satisfied: lxml in /opt/conda/lib/python3.6/site-packages (from pixiedust) (4.1.1)
    Requirement already satisfied: mpld3 in /opt/conda/lib/python3.6/site-packages (from pixiedust) (0.3)
    Requirement already satisfied: astunparse in /opt/conda/lib/python3.6/site-packages (from pixiedust) (1.6.2)
    Requirement already satisfied: geojson in /opt/conda/lib/python3.6/site-packages (from pixiedust) (2.5.0)
    Requirement already satisfied: requests in /opt/conda/lib/python3.6/site-packages (from pixiedust) (2.18.4)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /opt/conda/lib/python3.6/site-packages (from astunparse->pixiedust) (0.30.0)
    Requirement already satisfied: six<2.0,>=1.6.1 in /opt/conda/lib/python3.6/site-packages (from astunparse->pixiedust) (1.11.0)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/lib/python3.6/site-packages (from requests->pixiedust) (3.0.4)
    Requirement already satisfied: idna<2.7,>=2.5 in /opt/conda/lib/python3.6/site-packages (from requests->pixiedust) (2.6)
    Requirement already satisfied: urllib3<1.23,>=1.21.1 in /opt/conda/lib/python3.6/site-packages (from requests->pixiedust) (1.22)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.6/site-packages (from requests->pixiedust) (2017.11.5)



```python
import pixiedust
```

    Pixiedust database opened successfully




        <div style="margin:10px">
            <a href="https://github.com/ibm-watson-data-lab/pixiedust" target="_new">
                <img src="https://github.com/ibm-watson-data-lab/pixiedust/raw/master/docs/_static/pd_icon32.png" style="float:left;margin-right:10px"/>
            </a>
            <span>Pixiedust version 1.1.17</span>
        </div>
        



```python
pixiedust.optOut()
```

    Pixiedust will not collect anonymous install statistics.


# Load and Clean Dataset
In this workspace, the mini-dataset file is `mini_sparkify_event_data.json`. Load and clean the dataset, checking for invalid or missing data - for example, records without userids or sessionids. 


```python
path = "mini_sparkify_event_data.json"
user_log = spark.read.json(path)
```


```python
display(user_log)
```


```python
numerical_columns = [
 'itemInSession',

 'registration',
 'sessionId',
 'status',
 'ts']

categorical_columns = ['artist',
 'auth',
 'firstName',
 'gender',
 'lastName',
 'level',
 'location',
 'method',
 'page',
 'song',
 'userAgent',
 'userId']
```


```python
user_log.select(numerical_columns).describe().toPandas()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>summary</th>
      <th>itemInSession</th>
      <th>registration</th>
      <th>sessionId</th>
      <th>status</th>
      <th>ts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>count</td>
      <td>286500</td>
      <td>278154</td>
      <td>286500</td>
      <td>286500</td>
      <td>286500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mean</td>
      <td>114.41421291448516</td>
      <td>1.5353588340844272E12</td>
      <td>1041.526554973822</td>
      <td>210.05459685863875</td>
      <td>1.5409568898104834E12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stddev</td>
      <td>129.76726201140994</td>
      <td>3.291321616327586E9</td>
      <td>726.7762634630741</td>
      <td>31.50507848842214</td>
      <td>1.5075439608226302E9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>min</td>
      <td>0</td>
      <td>1521380675000</td>
      <td>1</td>
      <td>200</td>
      <td>1538352117000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>max</td>
      <td>1321</td>
      <td>1543247354000</td>
      <td>2474</td>
      <td>404</td>
      <td>1543799476000</td>
    </tr>
  </tbody>
</table>
</div>




```python
for column in categorical_columns:
    print(user_log.select(F.countDistinct(column).alias(column + " : distinct values")).show())
```

    +------------------------+
    |artist : distinct values|
    +------------------------+
    |                   17655|
    +------------------------+
    
    None
    +----------------------+
    |auth : distinct values|
    +----------------------+
    |                     4|
    +----------------------+
    
    None
    +---------------------------+
    |firstName : distinct values|
    +---------------------------+
    |                        189|
    +---------------------------+
    
    None
    +------------------------+
    |gender : distinct values|
    +------------------------+
    |                       2|
    +------------------------+
    
    None
    +--------------------------+
    |lastName : distinct values|
    +--------------------------+
    |                       173|
    +--------------------------+
    
    None
    +-----------------------+
    |level : distinct values|
    +-----------------------+
    |                      2|
    +-----------------------+
    
    None
    +--------------------------+
    |location : distinct values|
    +--------------------------+
    |                       114|
    +--------------------------+
    
    None
    +------------------------+
    |method : distinct values|
    +------------------------+
    |                       2|
    +------------------------+
    
    None
    +----------------------+
    |page : distinct values|
    +----------------------+
    |                    22|
    +----------------------+
    
    None
    +----------------------+
    |song : distinct values|
    +----------------------+
    |                 58480|
    +----------------------+
    
    None
    +---------------------------+
    |userAgent : distinct values|
    +---------------------------+
    |                         56|
    +---------------------------+
    
    None
    +------------------------+
    |userId : distinct values|
    +------------------------+
    |                     226|
    +------------------------+
    
    None



```python
# further analysis of relationship between page and status...
user_log.groupBy(["status", "method", "page"]).count().orderBy("status", "method", "count").show()
```

    +------+------+--------------------+------+
    |status|method|                page| count|
    +------+------+--------------------+------+
    |   200|   GET|            Register|    18|
    |   200|   GET|Cancellation Conf...|    52|
    |   200|   GET|             Upgrade|   499|
    |   200|   GET|               About|   924|
    |   200|   GET|            Settings|  1514|
    |   200|   GET|                Help|  1726|
    |   200|   GET|           Downgrade|  2055|
    |   200|   GET|         Roll Advert|  3933|
    |   200|   GET|                Home| 14457|
    |   200|   PUT|     Add to Playlist|  6526|
    |   200|   PUT|            NextSong|228108|
    |   307|   PUT| Submit Registration|     5|
    |   307|   PUT|              Cancel|    52|
    |   307|   PUT|    Submit Downgrade|    63|
    |   307|   PUT|      Submit Upgrade|   159|
    |   307|   PUT|       Save Settings|   310|
    |   307|   PUT|         Thumbs Down|  2546|
    |   307|   PUT|              Logout|  3226|
    |   307|   PUT|               Login|  3241|
    |   307|   PUT|          Add Friend|  4277|
    +------+------+--------------------+------+
    only showing top 20 rows
    



```python
# detailled analysis of feature "userAgent"
pd_df = user_log.groupBy("userAgent").count().orderBy("count").toPandas()
print(pd_df["userAgent"].tolist())
pd_df["count"].describe()
```

    ['Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/6.1.5 Safari/537.77.4"', 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)', 'Mozilla/5.0 (Windows NT 6.1; rv:31.0) Gecko/20100101 Firefox/31.0', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.74.9 (KHTML, like Gecko) Version/7.0.2 Safari/537.74.9"', 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.3 (KHTML, like Gecko) Version/8.0 Safari/600.1.3"', '"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_1 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D201 Safari/9537.53"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.76.4 (KHTML, like Gecko) Version/7.0.4 Safari/537.76.4"', '"Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36"', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; rv:31.0) Gecko/20100101 Firefox/31.0', 'Mozilla/5.0 (Windows NT 6.0; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (iPad; CPU OS 7_1_1 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D201 Safari/9537.53"', 'Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D167 Safari/9537.53"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:32.0) Gecko/20100101 Firefox/32.0', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko', '"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.8 (KHTML, like Gecko) Version/8.0 Safari/600.1.8"', 'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"', '"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.78.2 (KHTML, like Gecko) Version/7.0.6 Safari/537.78.2"', None, 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)', '"Mozilla/5.0 (iPad; CPU OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"']





    count       57.000000
    mean      5026.315789
    std       5570.658201
    min         62.000000
    25%       1262.000000
    50%       2544.000000
    75%       7624.000000
    max      22751.000000
    Name: count, dtype: float64



### Findings of descriptive statistics on data sample per feature: ###
* Summary per feature:
    * auth: no missing values; 4 distinct values. 
        * follow-up: check if missing values relate to value "Logged Out"
    * artist: few missing values; 17655 distinct values.
    * song: few missing values; 58480 distinct values.
    * userID: no missing values; 226 distinct values.
    * firstName: few missing values; 189 distinct values.
    * LastName: few missing values; 173 distinct values.
    * gender: few missing values; 2 distinct values.
    * ItemInSession: no missing values; 1322 distinct values.
        * follow-up: analyse meaning on example of one user
    * length: few missing values; 14865 distinct values.
        * follow-up: analyse meaning on example of one user
    * level: no missing values; 2 distinct values.
    * location: few missing values; 114 distinct values
        * follow-up: analyse user locations according to Country, State, etc
    * method: no missing values; 2 distinct values.
        * follow-up: analyse meaning on example of one user
    * page: no missing values; 22 distinct values.
    * registration: few missing values; 225 distinct values.
        * relates to 226 unique userID's minus empty name
        * format equals timestamp -> follow-up: convert to time/date
    * sessionID: few missing values; 2354 distinct values.
        * follow-up: analyse meaning on example of one user
    * status: no missing values; 3 distinct values.
    * ts: no missing values; 277447 unique values.
        * follow-up: conversion to date/ time
        * follow-up: analyse spread of date/ time
        * follow-up: further conversion to features for year/ month/ time
    * userAgent: few missing values; 56 distinct values
        * follow-up: further analyse values and value distribution

* Follow-up summary:
    1. "auth": check if missing values relate to value "Logged Out"
    2. analyse meaning on example of one user:
        * list = ["ItemInSession", "length", "method", "sessionID"]
    3. print categorical features with low cardinality (less than 56 distinct values):
        * list = ["auth", "gender", "level", "method", "page", "status", "userAgent"]
        * value name and value counts
    4. "location": analyse user locations and CSA (Combined Statistical Areas)
    * convert to time/date via new feature:
        * list = ["registration", "ts"]
    5. "ts": analyse spread of date/ time
    6. "ts": further conversion to features for year/ month/ time

#### Findings on (1):
* "auth" value "Cancelled" correlates to "page" value "Cancellation Confirmed".
* "auth" value "Guest" does not have any relevant information on users and can for churn use case be dropped.
    * follow-up: drop corresponding rows (user_log["auth"]=="Guest")
* "auth" value "Logged Out" does not have any relevant information on users and can for churn use case be dropped.
    * follow-up: drop corresponding rows (user_log["auth"]==""Logged Out"") 
* "auth" values "Logged In" has no missing artist values at page value "NextSong" (ca  72% in sample data set).
    * interpretation: users are in this cases listenting to music.
* "auth" values "Logged In" has missing artist values at all other pages (ca 18%in sample data set).
    * interpretation: users are in this cases not listenting to music, but doing other transactions


```python
# print missing values depending on column "auth" value
auth_values = user_log.select("auth").distinct().rdd.map(lambda r: r[0]).collect()
for value in auth_values:
    print("null values in rows with auth value {}:".format(value))
    value_df = user_log.filter(user_log["auth"] == value)
    missing_value_columns = ['artist',
                             'firstName',
                             'gender',
                             'lastName',
                             'length',
                             'location',
                             'registration',
                             'song',
                             'userAgent',
                             'userId',
                            ]
    for column in missing_value_columns:
        print("column {} : {}".format(column, value_df.filter(value_df[column].isNull()).count()))
```

#### (2) analyse meaning on example of one user: ####
list = ["ItemInSession", "length", "method", "sessionID"]

Findings:
* number of items per "sessionId" does not equal values in "ItemInSession"
* "method" appear to refer to user interaction. Interpretation:
    * PUT: action by user. Therefore PUT data should be furhter analyzed.
    * GET: reaction to user
* "lenght" refers to song length
* "sessionID": values are not unique. 
    * Several users can share the same session Id.
    * an examplary analysis showed not direct relation between users that share the same sessionId

#### Findings on (3) ####
* "status": refers to html status codes (https://en.wikipedia.org/wiki/List_of_HTTP_status_codes)
    * 200: page ok
    * 307: Temporary redirect
    * 404: error
* "userAgent": needs to be split up into further features for analysis, e.g. influence of operating system

### Remove values where "auth" value is either "Guest" or "Logged Out" since userId is missing


```python
# remove values where "auth" value is either "Guest" or "Logged Out" since userId is missing
user_log = user_log.filter(user_log["auth"].isin(*["Guest", "Logged Out"]) == False)
```

# Exploratory Data Analysis
When you're working with the full dataset, perform EDA by loading a small subset of the data and doing basic manipulations within Spark. In this workspace, you are already provided a small subset of data you can explore.

### Define Churn

Once you've done some preliminary analysis, create a column `Churn` to use as the label for your model. I suggest using the `Cancellation Confirmation` events to define your churn, which happen for both paid and free users. As a bonus task, you can also look into the `Downgrade` events.

### Explore Data
Once you've defined churn, perform some exploratory data analysis to observe the behavior for users who stayed vs users who churned. You can start by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.

## Define feature "churn"


```python
flag_churn_event = F.udf(lambda page: 1 if page == "Cancellation Confirmation" else 0, IntegerType())
user_log = user_log.withColumn("churn", flag_churn_event("page"))
```


```python
# analyze churn per userId
pd_df = user_log.select("userId", "churn").dropDuplicates().toPandas()
print("Distinct users in dataset: {}".format(pd_df["userId"].nunique()))
print("Value counts of churn over all userId's:")
print(pd_df["churn"].value_counts())
plt.hist(pd_df["churn"])
plt.show()
```

    Distinct users in dataset: 225
    Value counts of churn over all userId's:
    0    225
    1     52
    Name: churn, dtype: int64



![png](output_24_1.png)


## Explore Data

### "location": analyse user locations and CSA (Combined Statistical Areas)


```python
# create new feature CSA (Combined Statistical Areas) from location 
get_csa = F.split(user_log["location"], ", ")
user_log = user_log.withColumn("CSA", get_csa.getItem(1))
```


```python
# analyze CSA
pd_user_csa_churn = user_log.select("userId", "CSA", "churn").dropDuplicates().groupBy("userId", "CSA").agg(F.sum("churn").alias("churn")).toPandas()

pd_csa_analysis = pd_user_csa_churn.groupby("CSA", as_index=False).agg({"userId":"count", "churn": "sum"}).sort_values(by=["userId", "churn"], ascending = False)

plt.figure(figsize=(16, 16))
sns.barplot(x="userId", y="CSA" , data=pd_csa_analysis, color = "red");
sns.barplot(x="churn", y="CSA" , data=pd_csa_analysis, color = "blue");
```


![png](output_28_0.png)


####  (5) convert to timestamp "ts" time/date: ####


```python
def convert_ts_to_datetime(df, column):
    get_datetime = F.udf(lambda timestamp: datetime.datetime.fromtimestamp(timestamp/1000).isoformat())
    df = df.withColumn(column + "_ts", get_datetime(df[column]).cast(TimestampType()))
    return df
```


```python
# create new features in timestamp format from features "registration", "ts"
#column_list = ["registration", "ts"]
user_log = convert_ts_to_datetime(user_log, "ts")
```

#### (6) "ts": analyse spread of date/ time


```python
min_date, max_date = user_log.select(F.min("ts_ts"), F.max("ts_ts")).first()
print("Minimum and Maximum timestamp data:")
min_date, max_date
```

    Minimum and Maximum timestamp data:





    (datetime.datetime(2018, 10, 1, 0, 1, 57),
     datetime.datetime(2018, 12, 3, 1, 11, 16))



#### (7) "ts": further conversion to features for date/ hour


```python
# get new features day and hour
user_log = user_log.withColumn("ts_hour", F.hour("ts_ts"))
user_log = user_log.withColumn("ts_date", F.to_date("ts_ts"))
```


```python
print("Analyze log data over time:")
#pd_df = user_log.select(hour("ts_ts").alias("hour")).groupBy("hour").count().orderBy("hour").toPandas()
pd_df = user_log.select("ts_hour").groupBy("ts_hour").count().orderBy("ts_hour").toPandas()
pd_df.plot.line(x="ts_hour", y="count");
```

    Analyze log data over time:



![png](output_36_1.png)


## Features from "page" value ##

### Selection of page values for new features:
* "downgraded" from "Submit Downgrad"
* ...


```python
def create_page_value_feature(df, page_value, col_name):
    '''
    ARGS
    df: Spark dataframe
    page_value: categorical value in column "page" of df
    col_name: column name of new feature that is added to df
    
    OUTPUT: Spark dataframe with new column from page value
    
    Function that creates a new feature from a certain value of feature "page"
    '''
    flag_page_value_event = F.udf(lambda page: 1 if page == page_value else 0, IntegerType())
    return df.withColumn(col_name, flag_page_value_event("page"))
```


```python
page_value_feature_dict = {"Submit Downgrade" : "downgraded",
                           "Submit Upgrade" : "upgraded",
                           "Roll Advert" : "advert_shown",
                           "Thumbs Down": "thumps_down",
                           "Thumbs Up": "thumps_up",
                           "Add Friend": "friend_added",
                           "Add to Playlist" : "song_added"
                          }

for page_value in page_value_feature_dict.keys():
    column_name = page_value_feature_dict[page_value]
    user_log =  create_page_value_feature(user_log, page_value, column_name)
```

# Feature Engineering
Once you've familiarized yourself with the data, build out the features you find promising to train your model on. To work with the full dataset, you can follow the following steps.
- Write a script to extract the necessary features from the smaller subset of data
- Ensure that your script is scalable, using the best practices discussed in Lesson 3
- Try your script on the full data set, debugging your script if necessary

If you are working in the classroom workspace, you can just extract features based on the small subset of data contained here. Be sure to transfer over this work to the larger dataset when you work on your Spark cluster.

## Create new dataframe for features based on userId's


```python
# create new df for features
all_users_collect = user_log.select("userId").filter(user_log["userId"]!="").distinct().collect()
all_users = set([int(row["userId"]) for row in all_users_collect])
features_df = spark.createDataFrame(all_users, IntegerType()).withColumnRenamed('value', 'userId')
```

## Encode label "churned users"


```python
# create feature "label" for churned users
churned_users_collect = user_log.select("userId").filter(user_log["churn"]==1).collect()
churned_users = set([int(row["userId"]) for row in churned_users_collect])
get_churn = F.udf(lambda user: 1 if user in churned_users else 0, IntegerType())
features_df = features_df.withColumn("label", get_churn("userId"))
```

## Encode features "gender", "level"

### Encode "gender"
* gender value "M" = value 1
* gender value "F" = value 0


```python
# one hot encode gender in original df
one_hot_encode_gender = F.udf(lambda gender: 1 if gender == "M" else 0, IntegerType())
user_log = user_log.withColumn("gender_bin", one_hot_encode_gender("gender"))
```


```python
# join binary gender on userId in features df
user_gender_selection =  user_log.select(["userId", "gender_bin"]).dropDuplicates(subset=['userId'])
features_df = features_df.join(user_gender_selection, "userId")
```

### Encode "level"
* level value "paid" = value 1
* level value "free" = value 0


```python
# one hot encode level in original df
one_hot_encode_level = F.udf(lambda level: 1 if level == "paid" else 0, IntegerType())
user_log = user_log.withColumn("level_bin", one_hot_encode_level("level"))
```


```python
# get last state of level for each userId
user_level_selection =  user_log\
.select(["userId", "level_bin", "ts"])\
.orderBy("ts", ascending=False)\
.dropDuplicates(subset=['userId'])\
.drop("ts")

# join binary gender on userId in features df
features_df = features_df.join(user_level_selection, "userId")
```

## Encode non-binary page view features
* encode count of page view features per userId
* page view features to be included: 
['downgraded',
 'upgraded',
 'advert_shown',
 'thumps_down',
 'thumps_up',
 'friend_added',
 'song_added']


```python
page_features_count = user_log.groupBy("userId").sum('downgraded', 'upgraded',
 'advert_shown',
 'thumps_down',
 'thumps_up',
 'friend_added',
 'song_added')
features_df = features_df.join(page_features_count, "userId", how="left")
```

## Encode non-binary features related to service usage
* "song_count": songs per user
* "days_since_reg": days from registration until latest user timestamp of a user
* "session_time_seconds": accumulated session time per userId in seconds


```python
# create new feature "song_count" in features_df
song_count = user_log.groupBy("userId").agg(F.count("song").alias("song_count")).orderBy("song_count", ascending=False)
features_df = features_df.join(song_count, "userId", how="left")
```


```python
# create new feature "days_since_reg" in features_df

# create new features in timestamp format from features "registration"
reg_df=user_log.select("userId", "registration").filter(user_log["registration"].isNotNull()).distinct()
reg_df = convert_ts_to_datetime(reg_df, "registration")
reg_df = reg_df.withColumn("reg_date", F.to_date("registration_ts"))

# calculate difference between last user timestamp date and registration date
last_user_ts = user_log.groupBy("userId").agg(F.max("ts_date").alias("last_user_ts_date"))
reg_df = reg_df.join(last_user_ts, "userId", how="left")
reg_df = reg_df.withColumn("days_since_reg", F.datediff("last_user_ts_date", "reg_date"))

# add feature "days_since_reg" to features_df
features_df = features_df.join(reg_df.select("userId", "days_since_reg"), "userId", how="left")
```


```python
# get new feature for accumulated session time per userId

user_session_min_ts = user_log.groupBy("userId", "sessionID").agg(F.min("ts")).orderBy("userId", "sessionID")
user_session_max_ts = user_log.groupBy("userId", "sessionID").agg(F.max("ts")).orderBy("userId", "sessionID")
user_session_max_ts = user_session_max_ts.join(user_session_min_ts, ["userId", "sessionID"])
user_session_max_ts = user_session_max_ts.withColumn("session_time_seconds", ((user_session_max_ts["max(ts)"]-user_session_max_ts["min(ts)"])/1000).cast(LongType()))

# calculate total session time per user
total_user_session_time = user_session_max_ts.groupBy("userId").agg(F.sum("session_time_seconds").alias("total_session_time_sec"))

# add feature "session_time_seconds" to features_df
features_df = features_df.join(total_user_session_time, "userId", how="left")
```

## Explore features with regards to churn


```python
display(features_df)
```


```python
pd_features_df = features_df.toPandas()
```

### Pairplot of features


```python
# split features into two subsets to create a manageable pairplot
features_wo_id_label = pd_features_df.columns.values.tolist()[2:]
first_features_df = pd_features_df[features_wo_id_label[-6:]]
first_features_df["label"] = pd_features_df["label"]
second_features_df = pd_features_df[features_wo_id_label[:6]]
second_features_df["label"] = pd_features_df["label"]
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """



```python
# show relationship between variables for all features
plt.figure(figsize=(25, 25))
sns.pairplot(first_features_df, hue="label");
```


    <matplotlib.figure.Figure at 0x7f4fd9b359e8>



![png](output_64_1.png)



```python
# show relationship between variables for all features
plt.figure(figsize=(25, 25))
sns.pairplot(second_features_df, hue="label");
```


    <matplotlib.figure.Figure at 0x7f4fedab0400>



![png](output_65_1.png)



```python
# print correlation between variables
corr = pd_features_df.drop("userId", axis=1).corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True));
```


![png](output_66_0.png)


# START TMP: Save df_features to csv and load from it


```python
# save to csv 
features_df.toPandas().to_csv("features_pd_df.csv")
#features_df.write.save("features_sp_df.csv", format="csv", inferSchema=True, header="true")
```


```python
# load from csv
#features_df = spark.read.load("features_pd_df.csv", format="csv", inferSchema="true", header="true")
```

# +++ END TMP

# Feature selection

## Feature scaling and vectorization

### Vectorize and scale non-binary features
* Vectorization via VectorAssembler
* Scaling via MinMaxScaler
* user Pipeline to combine both in the transformation process


```python
nonbinary_feature_list = [
 'sum(downgraded)',
 'sum(upgraded)',
 'sum(advert_shown)',
 'sum(thumps_down)',
 'sum(thumps_up)',
 'sum(friend_added)',
 'sum(song_added)',
 'song_count',
 'days_since_reg',
 'total_session_time_sec']
```


```python
convert_vector_to_double = F.udf(lambda vector_value: round(float(list(vector_value)[0]),3), DoubleType())

for column in nonbinary_feature_list:
    # convert column to vector via VectorAssembler
    assembler = VectorAssembler(inputCols=[column], outputCol=column+"_vect")
    # Scale vectorized column
    scaler = MinMaxScaler(inputCol=column+"_vect", outputCol=column+"_scaled")
    # create Pipeline with assembler and scaler
    pipeline = Pipeline(stages=[assembler, scaler])
    # apply pipeline on features_df Dataframe
    features_df = pipeline.fit(features_df).transform(features_df) \
    .withColumn(column+"_scaled", convert_vector_to_double(column+"_scaled")).drop(column+"_vect")
```

### Merge scaled features to one feature vector


```python
# create feature list that shall be merged in on vector
feature_list = features_df.schema.names
# remove columns userId, label and all items in nonbinary_feature_list
remove_features_list= nonbinary_feature_list + ["userId", "label"]
feature_list = [item for item in feature_list if item not in remove_features_list]
# assemble features in feature_list to one vector using VectorAssembler
assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
features_df = assembler.transform(features_df)
```

## Perform PCA to select relevant features


```python
pca_number = 3
pca = PCA(k=pca_number, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(features_df.select("features"))

#result = model.transform(features_df.select("features")).select("pcaFeatures")
#result.show(truncate=False)
#features_df = features_df.withColumn("pcaFeatures", model.transform(features_df.select("features")).select("pcaFeatures"))
pca_features = model.transform(features_df.select("features")).select("pcaFeatures")

# join column "pcaFeatures" to existing dataframe
pca_features = pca_features.withColumn("id", F.monotonically_increasing_id())
features_df = features_df.withColumn("id", F.monotonically_increasing_id())
features_df = features_df.join(pca_features, "id", "outer").drop("id")
```


```python
print("Explained variance by {} principal components: {:.2f}%".format(pca_number, sum(model.explainedVariance)*100))
```

    Explained variance by 3 principal components: 88.48%


# Modeling
Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. Evaluate the accuracy of the various models, tuning parameters as necessary. Determine your winning model based on test accuracy and report results on the validation set. Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.

## Split in training, test, validation set


```python
train, test = features_df.randomSplit([0.8, 0.2], seed=42)


plt.hist(features_df.toPandas()['label'])
plt.show()
```


![png](output_83_0.png)


### Analyze label class imbalance


```python
# calculate balancing ratio for account for class imbalance
balancing_ratio = train.filter(train['label']==0).count()/train.count()
train=train.withColumn("classWeights", F.when(train.label == 1,balancing_ratio).otherwise(1-balancing_ratio))
```

## Machine Learning Model Selection, Tuning and Evaluation
 * Model learning problem category: supervised learning, logistic regression
 * ML estimators from pyspark.ml:
     * LogisticRegression
 * ML hyperparameters in estimators (for grid search/ tuning):
     * LogisticRegression(maxIter=..., regParam=..., elasticNetParam=...)
 * ML evaluators from pyspark.ml:
     * BinaryClassificationEvaluator


```python
# Create a logistic regression object
lr = LogisticRegression(featuresCol = 'pcaFeatures', labelCol = 'label', weightCol="classWeights")

# create evaluator
evaluator = BinaryClassificationEvaluator(metricName = 'areaUnderPR')
```


```python
# create lr_model
#lr_model = lr.fit(train)
#training_summary = lr_model.summary
```


```python
# ToDo: evaluate training summary ++++++++++++++++

# TBD
```

## Tune Model
* use cross validation via CrossValidator and paramGrid


```python
# tune model via CrossValidator and parameter Grid 

# build paramGrid
paramGrid = (ParamGridBuilder() \
    .addGrid(lr.maxIter, [1, 5, 10]) \
    .addGrid(lr.regParam,[0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build())

# build cross validator
crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)
```


```python
# run cross validation
# train model on train data
crossval_model = crossval.fit(train)
# predict on test data
pred = crossval_model.transform(test)
```


```python
#cvModel_q1 = crossval.fit(training)
```


```python
#cvModel_q1.avgMetrics
```


```python
#results = cvModel_q1.transform(test)
```

## Evaluate results
* use scikit learn metrics f1, precision, recall for model evaluation


```python
# evaluate results
pd_pred = pred.toPandas()
```


```python
# show resulting format
pd_pred.head(3)
```


```python
# calculate score for f1, precision, recall
f1 = f1_score(pd_pred.label, pd_pred.prediction)
recall = recall_score(pd_pred.label, pd_pred.prediction)
precision = precision_score(pd_pred.label, pd_pred.prediction)
```


```python
print("F1 Score: {:.2f}, Recall: {:.2f}, Precision: {:.2f}".format(f1, recall, precision))
```

# Final Steps
Clean up your code, adding comments and renaming variables to make the code easier to read and maintain. Refer to the Spark Project Overview page and Data Scientist Capstone Project Rubric to make sure you are including all components of the capstone project and meet all expectations. Remember, this includes thorough documentation in a README file in a Github repository, as well as a web app or blog post.


```python

```
