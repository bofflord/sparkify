
# Sparkify Project Workspace
This workspace contains a tiny subset (128MB) of the full dataset available (12GB). Feel free to use this workspace to build your project, or to explore a smaller subset with Spark before deploying your cluster on the cloud. Instructions for setting up your Spark cluster is included in the last lesson of the Extracurricular Spark Course content.

You can follow the steps below to guide your data analysis and model building portion of this project.


```python
# Install additional libraries via pip in the current Jupyter kernel
import sys
!{sys.executable} -m pip install pixiedust
```

    Collecting pixiedust
    [?25l  Downloading https://files.pythonhosted.org/packages/bc/a8/e84b2ed12ee387589c099734b6f914a520e1fef2733c955982623080e813/pixiedust-1.1.17.tar.gz (197kB)
    [K    100% |████████████████████████████████| 204kB 29.2MB/s a 0:00:01
    [?25hCollecting mpld3 (from pixiedust)
    [?25l  Downloading https://files.pythonhosted.org/packages/91/95/a52d3a83d0a29ba0d6898f6727e9858fe7a43f6c2ce81a5fe7e05f0f4912/mpld3-0.3.tar.gz (788kB)
    [K    100% |████████████████████████████████| 798kB 8.9MB/s eta 0:00:01    72% |███████████████████████▎        | 573kB 17.8MB/s eta 0:00:01
    [?25hRequirement already satisfied: lxml in /opt/conda/lib/python3.6/site-packages (from pixiedust) (4.1.1)
    Collecting geojson (from pixiedust)
      Downloading https://files.pythonhosted.org/packages/e4/8d/9e28e9af95739e6d2d2f8d4bef0b3432da40b7c3588fbad4298c1be09e48/geojson-2.5.0-py2.py3-none-any.whl
    Collecting astunparse (from pixiedust)
      Downloading https://files.pythonhosted.org/packages/2e/37/5dd0dd89b87bb5f0f32a7e775458412c52d78f230ab8d0c65df6aabc4479/astunparse-1.6.2-py2.py3-none-any.whl
    Requirement already satisfied: markdown in /opt/conda/lib/python3.6/site-packages (from pixiedust) (2.6.9)
    Requirement already satisfied: colour in /opt/conda/lib/python3.6/site-packages (from pixiedust) (0.1.5)
    Requirement already satisfied: requests in /opt/conda/lib/python3.6/site-packages (from pixiedust) (2.18.4)
    Requirement already satisfied: six<2.0,>=1.6.1 in /opt/conda/lib/python3.6/site-packages (from astunparse->pixiedust) (1.11.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /opt/conda/lib/python3.6/site-packages (from astunparse->pixiedust) (0.30.0)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/lib/python3.6/site-packages (from requests->pixiedust) (3.0.4)
    Requirement already satisfied: idna<2.7,>=2.5 in /opt/conda/lib/python3.6/site-packages (from requests->pixiedust) (2.6)
    Requirement already satisfied: urllib3<1.23,>=1.21.1 in /opt/conda/lib/python3.6/site-packages (from requests->pixiedust) (1.22)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.6/site-packages (from requests->pixiedust) (2017.11.5)
    Building wheels for collected packages: pixiedust, mpld3
      Running setup.py bdist_wheel for pixiedust ... [?25ldone
    [?25h  Stored in directory: /root/.cache/pip/wheels/25/fa/a5/09c1e8f4c91b34c5f7f4ac6e41be81dd0667030a2372546a8d
      Running setup.py bdist_wheel for mpld3 ... [?25ldone
    [?25h  Stored in directory: /root/.cache/pip/wheels/c0/47/fb/8a64f89aecfe0059830479308ad42d62e898a3e3cefdf6ba28
    Successfully built pixiedust mpld3
    Installing collected packages: mpld3, geojson, astunparse, pixiedust
    Successfully installed astunparse-1.6.2 geojson-2.5.0 mpld3-0.3 pixiedust-1.1.17



```python
# import libraries
from pyspark.sql import SparkSession, Window

from pyspark.sql.functions import udf, sum as Fsum, desc, asc, countDistinct, col, to_date, year, month, dayofmonth, minute, hour, datediff, min, max, isnull

from pyspark.sql.types import IntegerType, DateType, TimestampType, StringType

import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt

import datetime

import pixiedust

```

    Pixiedust database opened successfully
    Table VERSION_TRACKER created successfully
    Table METRICS_TRACKER created successfully
    
    Share anonymous install statistics? (opt-out instructions)
    
    PixieDust will record metadata on its environment the next time the package is installed or updated. The data is anonymized and aggregated to help plan for future releases, and records only the following values:
    
    {
       "data_sent": currentDate,
       "runtime": "python",
       "application_version": currentPixiedustVersion,
       "space_id": nonIdentifyingUniqueId,
       "config": {
           "repository_id": "https://github.com/ibm-watson-data-lab/pixiedust",
           "target_runtimes": ["Data Science Experience"],
           "event_id": "web",
           "event_organizer": "dev-journeys"
       }
    }
    You can opt out by calling pixiedust.optOut() in a new cell.




        <div style="margin:10px">
            <a href="https://github.com/ibm-watson-data-lab/pixiedust" target="_new">
                <img src="https://github.com/ibm-watson-data-lab/pixiedust/raw/master/docs/_static/pd_icon32.png" style="float:left;margin-right:10px"/>
            </a>
            <span>Pixiedust version 1.1.17</span>
        </div>
        


    [31mPixiedust runtime updated. Please restart kernel[0m
    Table USER_PREFERENCES created successfully
    Table service_connections created successfully



```python
pixiedust.optOut()
```

    Pixiedust will not collect anonymous install statistics.



```python
# create a Spark session
spark = SparkSession \
    .builder \
    .appName("Sparkify") \
    .getOrCreate()
```

# Load and Clean Dataset
In this workspace, the mini-dataset file is `mini_sparkify_event_data.json`. Load and clean the dataset, checking for invalid or missing data - for example, records without userids or sessionids. 


```python
path = "mini_sparkify_event_data.json"
user_log = spark.read.json(path)
```

# Exploratory Data Analysis
When you're working with the full dataset, perform EDA by loading a small subset of the data and doing basic manipulations within Spark. In this workspace, you are already provided a small subset of data you can explore.

### Define Churn

Once you've done some preliminary analysis, create a column `Churn` to use as the label for your model. I suggest using the `Cancellation Confirmation` events to define your churn, which happen for both paid and free users. As a bonus task, you can also look into the `Downgrade` events.

### Explore Data
Once you've defined churn, perform some exploratory data analysis to observe the behavior for users who stayed vs users who churned. You can start by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.

#### Exploratory Data Analysis


```python
user_log.printSchema()
```

    root
     |-- artist: string (nullable = true)
     |-- auth: string (nullable = true)
     |-- firstName: string (nullable = true)
     |-- gender: string (nullable = true)
     |-- itemInSession: long (nullable = true)
     |-- lastName: string (nullable = true)
     |-- length: double (nullable = true)
     |-- level: string (nullable = true)
     |-- location: string (nullable = true)
     |-- method: string (nullable = true)
     |-- page: string (nullable = true)
     |-- registration: long (nullable = true)
     |-- sessionId: long (nullable = true)
     |-- song: string (nullable = true)
     |-- status: long (nullable = true)
     |-- ts: long (nullable = true)
     |-- userAgent: string (nullable = true)
     |-- userId: string (nullable = true)
    



```python
print(user_log.take(1))
```

    [Row(artist='Martha Tilston', auth='Logged In', firstName='Colin', gender='M', itemInSession=50, lastName='Freeman', length=277.89016, level='paid', location='Bakersfield, CA', method='PUT', page='NextSong', registration=1538173362000, sessionId=29, song='Rockpools', status=200, ts=1538352117000, userAgent='Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', userId='30')]



```python
print("Number of rows in dataset: {}".format(user_log.count())) 
```

    Number of rows in dataset: 286500



```python
for column in user_log.columns:
    print("Analysis of column {}".format(column))
    print("Statistical properties:")
    print(user_log.describe(column).show())
    print("\nCount of unique values in column:")
    print(user_log.select(countDistinct(column)).show(),"\n")
```

    Analysis of column artist
    Statistical properties:
    +-------+------------------+
    |summary|            artist|
    +-------+------------------+
    |  count|            228108|
    |   mean| 551.0852017937219|
    | stddev|1217.7693079161374|
    |    min|               !!!|
    |    max| ÃÂlafur Arnalds|
    +-------+------------------+
    
    None
    
    Count of unique values in column:
    +----------------------+
    |count(DISTINCT artist)|
    +----------------------+
    |                 17655|
    +----------------------+
    
    None 
    
    Analysis of column auth
    Statistical properties:
    +-------+----------+
    |summary|      auth|
    +-------+----------+
    |  count|    286500|
    |   mean|      null|
    | stddev|      null|
    |    min| Cancelled|
    |    max|Logged Out|
    +-------+----------+
    
    None
    
    Count of unique values in column:
    +--------------------+
    |count(DISTINCT auth)|
    +--------------------+
    |                   4|
    +--------------------+
    
    None 
    
    Analysis of column firstName
    Statistical properties:
    +-------+---------+
    |summary|firstName|
    +-------+---------+
    |  count|   278154|
    |   mean|     null|
    | stddev|     null|
    |    min| Adelaida|
    |    max|   Zyonna|
    +-------+---------+
    
    None
    
    Count of unique values in column:
    +-------------------------+
    |count(DISTINCT firstName)|
    +-------------------------+
    |                      189|
    +-------------------------+
    
    None 
    
    Analysis of column gender
    Statistical properties:
    +-------+------+
    |summary|gender|
    +-------+------+
    |  count|278154|
    |   mean|  null|
    | stddev|  null|
    |    min|     F|
    |    max|     M|
    +-------+------+
    
    None
    
    Count of unique values in column:
    +----------------------+
    |count(DISTINCT gender)|
    +----------------------+
    |                     2|
    +----------------------+
    
    None 
    
    Analysis of column itemInSession
    Statistical properties:
    +-------+------------------+
    |summary|     itemInSession|
    +-------+------------------+
    |  count|            286500|
    |   mean|114.41421291448516|
    | stddev|129.76726201140994|
    |    min|                 0|
    |    max|              1321|
    +-------+------------------+
    
    None
    
    Count of unique values in column:
    +-----------------------------+
    |count(DISTINCT itemInSession)|
    +-----------------------------+
    |                         1322|
    +-----------------------------+
    
    None 
    
    Analysis of column lastName
    Statistical properties:
    +-------+--------+
    |summary|lastName|
    +-------+--------+
    |  count|  278154|
    |   mean|    null|
    | stddev|    null|
    |    min|   Adams|
    |    max|  Wright|
    +-------+--------+
    
    None
    
    Count of unique values in column:
    +------------------------+
    |count(DISTINCT lastName)|
    +------------------------+
    |                     173|
    +------------------------+
    
    None 
    
    Analysis of column length
    Statistical properties:
    +-------+-----------------+
    |summary|           length|
    +-------+-----------------+
    |  count|           228108|
    |   mean|249.1171819778458|
    | stddev|99.23517921058361|
    |    min|          0.78322|
    |    max|       3024.66567|
    +-------+-----------------+
    
    None
    
    Count of unique values in column:
    +----------------------+
    |count(DISTINCT length)|
    +----------------------+
    |                 14865|
    +----------------------+
    
    None 
    
    Analysis of column level
    Statistical properties:
    +-------+------+
    |summary| level|
    +-------+------+
    |  count|286500|
    |   mean|  null|
    | stddev|  null|
    |    min|  free|
    |    max|  paid|
    +-------+------+
    
    None
    
    Count of unique values in column:
    +---------------------+
    |count(DISTINCT level)|
    +---------------------+
    |                    2|
    +---------------------+
    
    None 
    
    Analysis of column location
    Statistical properties:
    +-------+-----------------+
    |summary|         location|
    +-------+-----------------+
    |  count|           278154|
    |   mean|             null|
    | stddev|             null|
    |    min|       Albany, OR|
    |    max|Winston-Salem, NC|
    +-------+-----------------+
    
    None
    
    Count of unique values in column:
    +------------------------+
    |count(DISTINCT location)|
    +------------------------+
    |                     114|
    +------------------------+
    
    None 
    
    Analysis of column method
    Statistical properties:
    +-------+------+
    |summary|method|
    +-------+------+
    |  count|286500|
    |   mean|  null|
    | stddev|  null|
    |    min|   GET|
    |    max|   PUT|
    +-------+------+
    
    None
    
    Count of unique values in column:
    +----------------------+
    |count(DISTINCT method)|
    +----------------------+
    |                     2|
    +----------------------+
    
    None 
    
    Analysis of column page
    Statistical properties:
    +-------+-------+
    |summary|   page|
    +-------+-------+
    |  count| 286500|
    |   mean|   null|
    | stddev|   null|
    |    min|  About|
    |    max|Upgrade|
    +-------+-------+
    
    None
    
    Count of unique values in column:
    +--------------------+
    |count(DISTINCT page)|
    +--------------------+
    |                  22|
    +--------------------+
    
    None 
    
    Analysis of column registration
    Statistical properties:
    +-------+--------------------+
    |summary|        registration|
    +-------+--------------------+
    |  count|              278154|
    |   mean|1.535358834084427...|
    | stddev| 3.291321616327586E9|
    |    min|       1521380675000|
    |    max|       1543247354000|
    +-------+--------------------+
    
    None
    
    Count of unique values in column:
    +----------------------------+
    |count(DISTINCT registration)|
    +----------------------------+
    |                         225|
    +----------------------------+
    
    None 
    
    Analysis of column sessionId
    Statistical properties:
    +-------+-----------------+
    |summary|        sessionId|
    +-------+-----------------+
    |  count|           286500|
    |   mean|1041.526554973822|
    | stddev|726.7762634630741|
    |    min|                1|
    |    max|             2474|
    +-------+-----------------+
    
    None
    
    Count of unique values in column:
    +-------------------------+
    |count(DISTINCT sessionId)|
    +-------------------------+
    |                     2354|
    +-------------------------+
    
    None 
    
    Analysis of column song
    Statistical properties:
    +-------+--------------------+
    |summary|                song|
    +-------+--------------------+
    |  count|              228108|
    |   mean|            Infinity|
    | stddev|                 NaN|
    |    min|ÃÂg ÃÂtti Gr...|
    |    max|ÃÂau hafa slopp...|
    +-------+--------------------+
    
    None
    
    Count of unique values in column:
    +--------------------+
    |count(DISTINCT song)|
    +--------------------+
    |               58480|
    +--------------------+
    
    None 
    
    Analysis of column status
    Statistical properties:
    +-------+------------------+
    |summary|            status|
    +-------+------------------+
    |  count|            286500|
    |   mean|210.05459685863875|
    | stddev| 31.50507848842214|
    |    min|               200|
    |    max|               404|
    +-------+------------------+
    
    None
    
    Count of unique values in column:
    +----------------------+
    |count(DISTINCT status)|
    +----------------------+
    |                     3|
    +----------------------+
    
    None 
    
    Analysis of column ts
    Statistical properties:
    +-------+--------------------+
    |summary|                  ts|
    +-------+--------------------+
    |  count|              286500|
    |   mean|1.540956889810483...|
    | stddev|1.5075439608226302E9|
    |    min|       1538352117000|
    |    max|       1543799476000|
    +-------+--------------------+
    
    None
    
    Count of unique values in column:
    +------------------+
    |count(DISTINCT ts)|
    +------------------+
    |            277447|
    +------------------+
    
    None 
    
    Analysis of column userAgent
    Statistical properties:
    +-------+--------------------+
    |summary|           userAgent|
    +-------+--------------------+
    |  count|              278154|
    |   mean|                null|
    | stddev|                null|
    |    min|"Mozilla/5.0 (Mac...|
    |    max|Mozilla/5.0 (comp...|
    +-------+--------------------+
    
    None
    
    Count of unique values in column:
    +-------------------------+
    |count(DISTINCT userAgent)|
    +-------------------------+
    |                       56|
    +-------------------------+
    
    None 
    
    Analysis of column userId
    Statistical properties:
    +-------+-----------------+
    |summary|           userId|
    +-------+-----------------+
    |  count|           286500|
    |   mean|59682.02278593872|
    | stddev|109091.9499991047|
    |    min|                 |
    |    max|               99|
    +-------+-----------------+
    
    None
    
    Count of unique values in column:
    +----------------------+
    |count(DISTINCT userId)|
    +----------------------+
    |                   226|
    +----------------------+
    
    None 
    


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
    4. "location": analyse user locations according to Country, State, etc
    * convert to time/date:
        * list = ["registration", "ts"]
    5. "ts": analyse spread of date/ time
    6. "ts": further conversion to features for year/ month/ time

#### (1) "auth": check if missing values relate to value "Logged Out"


```python
# (1) check if missing values relate to value "Logged Out"
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
                             'userAgent']
    for column in missing_value_columns:
        print("column {} : {}".format(column, value_df.filter(value_df[column].isNull()).count()))
```

    null values in rows with auth value Logged Out:
    column artist : 8249
    column firstName : 8249
    column gender : 8249
    column lastName : 8249
    column length : 8249
    column location : 8249
    column registration : 8249
    column song : 8249
    column userAgent : 8249
    null values in rows with auth value Cancelled:
    column artist : 52
    column firstName : 0
    column gender : 0
    column lastName : 0
    column length : 52
    column location : 0
    column registration : 0
    column song : 52
    column userAgent : 0
    null values in rows with auth value Guest:
    column artist : 97
    column firstName : 97
    column gender : 97
    column lastName : 97
    column length : 97
    column location : 97
    column registration : 97
    column song : 97
    column userAgent : 97
    null values in rows with auth value Logged In:
    column artist : 49994
    column firstName : 0
    column gender : 0
    column lastName : 0
    column length : 49994
    column location : 0
    column registration : 0
    column song : 49994
    column userAgent : 0



```python
print("Value counts of distinct values in feature auth:")
user_log.groupBy("auth").count().orderBy("count").show()
```

    Value counts of distinct values in feature auth:
    +----------+------+
    |      auth| count|
    +----------+------+
    | Cancelled|    52|
    |     Guest|    97|
    |Logged Out|  8249|
    | Logged In|278102|
    +----------+------+
    



```python
print("Analyse page counts where users are logged in and feature artist has missing values:")
user_log.filter((user_log["auth"]=="Logged In")&(user_log["artist"].isNull())).groupBy("page").count().orderBy("count").show()
```

    Analyse page counts where users are logged in and feature artist has missing values:
    +----------------+-----+
    |            page|count|
    +----------------+-----+
    |          Cancel|   52|
    |Submit Downgrade|   63|
    |  Submit Upgrade|  159|
    |           Error|  252|
    |   Save Settings|  310|
    |           About|  495|
    |         Upgrade|  499|
    |            Help| 1454|
    |        Settings| 1514|
    |       Downgrade| 2055|
    |     Thumbs Down| 2546|
    |          Logout| 3226|
    |     Roll Advert| 3933|
    |      Add Friend| 4277|
    | Add to Playlist| 6526|
    |            Home|10082|
    |       Thumbs Up|12551|
    +----------------+-----+
    



```python
print("Analyse page counts where users are logged in and feature artist has NO missing values:")
user_log.filter((user_log["auth"]=="Logged In")&(user_log["artist"].isNotNull())).groupBy("page").count().orderBy("count").show()
```

    Analyse page counts where users are logged in and feature artist has NO missing values:
    +--------+------+
    |    page| count|
    +--------+------+
    |NextSong|228108|
    +--------+------+
    



```python
print("Analyse value count of Roll Advert depending on level value:")
user_log_valid.filter(user_log_valid["page"]=="Roll Advert").groupBy("level").count().show()
```

    Analyse value count of Roll Advert depending on level value:
    +-----+-----+
    |level|count|
    +-----+-----+
    | free| 3687|
    | paid|  246|
    +-----+-----+
    


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

## Remove auth values "Guest" and "Logged Out" from Dataset


```python
user_log_valid = user_log.filter(user_log["auth"].isin(*["Guest", "Logged Out"]) == False)
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


```python
# Analyze feature method
pd_method_df = user_log_valid.groupBy(["method","page"]).count().orderBy(["method","count"]).toPandas()
pd_method_df
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
      <th>method</th>
      <th>page</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GET</td>
      <td>Cancellation Confirmation</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GET</td>
      <td>Error</td>
      <td>252</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GET</td>
      <td>About</td>
      <td>495</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GET</td>
      <td>Upgrade</td>
      <td>499</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GET</td>
      <td>Help</td>
      <td>1454</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GET</td>
      <td>Settings</td>
      <td>1514</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GET</td>
      <td>Downgrade</td>
      <td>2055</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GET</td>
      <td>Roll Advert</td>
      <td>3933</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GET</td>
      <td>Home</td>
      <td>10082</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PUT</td>
      <td>Cancel</td>
      <td>52</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PUT</td>
      <td>Submit Downgrade</td>
      <td>63</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PUT</td>
      <td>Submit Upgrade</td>
      <td>159</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PUT</td>
      <td>Save Settings</td>
      <td>310</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PUT</td>
      <td>Thumbs Down</td>
      <td>2546</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PUT</td>
      <td>Logout</td>
      <td>3226</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PUT</td>
      <td>Add Friend</td>
      <td>4277</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PUT</td>
      <td>Add to Playlist</td>
      <td>6526</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PUT</td>
      <td>Thumbs Up</td>
      <td>12551</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PUT</td>
      <td>NextSong</td>
      <td>228108</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.scatter(pd_method_df["method"], pd_method_df["page"].sort_index(ascending=False), c=pd_method_df["count"], cmap='viridis')
plt.xlabel("method")
plt.ylabel("page");
```


![png](output_25_0.png)



```python
# analyze length
user_log_valid.select(["song", "length"]).distinct().take(5)
```




    [Row(song="She's Mine", length=271.51628),
     Row(song='Nossa Senhora Do Tejo', length=287.4771),
     Row(song='So Lonely', length=287.81669),
     Row(song='Face Down (Album Version)', length=191.84281),
     Row(song='Lonely Summer Nights', length=199.13098)]




```python
# analyze sessionId
print("Maximum value count for one Id in Session Id:")
print(user_log_valid.groupBy("sessionId").count().orderBy("count").agg({"count": "max"}).collect()[0][0])
pd_sessionid_df = user_log_valid.groupBy("sessionId").count().orderBy("count").toPandas()
pd_sessionid_df.describe()
```

    Maximum value count for one Id in Session Id:
    1282





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
      <th>sessionId</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2312.000000</td>
      <td>2312.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1183.699394</td>
      <td>120.308824</td>
    </tr>
    <tr>
      <th>std</th>
      <td>690.855976</td>
      <td>134.407683</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>584.750000</td>
      <td>28.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1177.500000</td>
      <td>72.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1776.250000</td>
      <td>169.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2474.000000</td>
      <td>1282.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Analyze count of unique userId's per sessionId
print("Count of unique userId's per sessionId:")
user_log_valid.groupBy(["sessionId", "userId"]).count().orderBy(["sessionId", "userId"]).groupBy("sessionId").count().orderBy("count").describe().show()
```

    Count of unique userId's per sessionId:
    +-------+------------------+------------------+
    |summary|         sessionId|             count|
    +-------+------------------+------------------+
    |  count|              2312|              2312|
    |   mean|1183.6993944636679|1.3737024221453287|
    | stddev| 690.8559764013388|0.8037991538528159|
    |    min|                 1|                 1|
    |    max|              2474|                 4|
    +-------+------------------+------------------+
    



```python
# convert to pandas series for further analysis
pd_usercount_per_sessionid_df = user_log_valid.groupBy(["sessionId", "userId"]).count().orderBy(["sessionId", "userId"]).groupBy("sessionId").count().orderBy("count").toPandas()
```


```python
# get examples of sessionId's that share same Id for 4 users
pd_usercount_per_sessionid_df[pd_usercount_per_sessionid_df["count"]==4].head()
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
      <th>sessionId</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2234</th>
      <td>65</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>54</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>112</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>113</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>167</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# analyze sessionId 65 further...
user_log_valid.filter(user_log_valid["sessionId"]==65).groupBy("userId").count().orderBy("count").show()
```

    +------+-----+
    |userId|count|
    +------+-----+
    |200021|   10|
    |    66|   15|
    |100013|   73|
    |300023|  270|
    +------+-----+
    



```python
# select userId's "200021", "66" for further analysis

pd_df = user_log_valid.filter((user_log_valid["sessionId"]==65) & (user_log_valid["userId"].isin(*["200021", "66"]))).toPandas()
pd_df
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
      <th>artist</th>
      <th>auth</th>
      <th>firstName</th>
      <th>gender</th>
      <th>itemInSession</th>
      <th>lastName</th>
      <th>length</th>
      <th>level</th>
      <th>location</th>
      <th>method</th>
      <th>page</th>
      <th>registration</th>
      <th>sessionId</th>
      <th>song</th>
      <th>status</th>
      <th>ts</th>
      <th>userAgent</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bersuit Vergarabat</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>0</td>
      <td>Johnston</td>
      <td>236.98240</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>La logia (iambo-iombo)</td>
      <td>200</td>
      <td>1538682135000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>1</td>
      <td>Johnston</td>
      <td>NaN</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>Logout</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>None</td>
      <td>307</td>
      <td>1538682136000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>4</td>
      <td>Johnston</td>
      <td>NaN</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>GET</td>
      <td>Home</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>None</td>
      <td>200</td>
      <td>1538682221000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Queens Of The Stone Age</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>5</td>
      <td>Johnston</td>
      <td>192.46975</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>You Think I Ain't Worth A Dollar_ But I Feel L...</td>
      <td>200</td>
      <td>1538682371000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>4</th>
      <td>She &amp; Him</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>6</td>
      <td>Johnston</td>
      <td>153.52118</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>Why Do You Let Me Stay Here?</td>
      <td>200</td>
      <td>1538682563000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bumblefoot</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>7</td>
      <td>Johnston</td>
      <td>246.77832</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>Green</td>
      <td>200</td>
      <td>1538682716000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Madlib The Beat Konducta</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>8</td>
      <td>Johnston</td>
      <td>92.29016</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>Life</td>
      <td>200</td>
      <td>1538682962000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Evanescence</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>9</td>
      <td>Johnston</td>
      <td>247.35302</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>My Last Breath</td>
      <td>200</td>
      <td>1538683054000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Babyshambles</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>10</td>
      <td>Johnston</td>
      <td>214.88281</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>There She Goes</td>
      <td>200</td>
      <td>1538683301000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dilated Peoples</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>11</td>
      <td>Johnston</td>
      <td>236.59057</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>World On Wheels</td>
      <td>200</td>
      <td>1538683515000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Dr Rubber Funk</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>12</td>
      <td>Johnston</td>
      <td>252.05506</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>Bossa For The Devil</td>
      <td>200</td>
      <td>1538683751000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Bell X1</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>13</td>
      <td>Johnston</td>
      <td>233.29914</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>Flame</td>
      <td>200</td>
      <td>1538684003000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Alliance Ethnik</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>14</td>
      <td>Johnston</td>
      <td>252.21179</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>ReprÃÂ©sente</td>
      <td>200</td>
      <td>1538684236000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Joshua Radin</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>15</td>
      <td>Johnston</td>
      <td>202.44853</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>No Envy No Fear</td>
      <td>200</td>
      <td>1538684488000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>14</th>
      <td>None</td>
      <td>Logged In</td>
      <td>Alyssa</td>
      <td>F</td>
      <td>16</td>
      <td>Johnston</td>
      <td>NaN</td>
      <td>free</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>PUT</td>
      <td>Add to Playlist</td>
      <td>1532634173000</td>
      <td>65</td>
      <td>None</td>
      <td>200</td>
      <td>1538684503000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Matt Wertz</td>
      <td>Logged In</td>
      <td>Liliana</td>
      <td>F</td>
      <td>0</td>
      <td>Owen</td>
      <td>236.93016</td>
      <td>free</td>
      <td>Detroit-Warren-Dearborn, MI</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1535032914000</td>
      <td>65</td>
      <td>Counting to 100</td>
      <td>200</td>
      <td>1538717365000</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; r...</td>
      <td>200021</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Sex Pistols</td>
      <td>Logged In</td>
      <td>Liliana</td>
      <td>F</td>
      <td>1</td>
      <td>Owen</td>
      <td>188.91710</td>
      <td>free</td>
      <td>Detroit-Warren-Dearborn, MI</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1535032914000</td>
      <td>65</td>
      <td>EMI (Unlimited Edition)</td>
      <td>200</td>
      <td>1538717601000</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; r...</td>
      <td>200021</td>
    </tr>
    <tr>
      <th>17</th>
      <td>None</td>
      <td>Logged In</td>
      <td>Liliana</td>
      <td>F</td>
      <td>2</td>
      <td>Owen</td>
      <td>NaN</td>
      <td>free</td>
      <td>Detroit-Warren-Dearborn, MI</td>
      <td>PUT</td>
      <td>Thumbs Down</td>
      <td>1535032914000</td>
      <td>65</td>
      <td>None</td>
      <td>307</td>
      <td>1538717602000</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; r...</td>
      <td>200021</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Lonnie Gordon</td>
      <td>Logged In</td>
      <td>Liliana</td>
      <td>F</td>
      <td>3</td>
      <td>Owen</td>
      <td>181.21098</td>
      <td>free</td>
      <td>Detroit-Warren-Dearborn, MI</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1535032914000</td>
      <td>65</td>
      <td>Catch You Baby (Steve Pitron &amp; Max Sanna Radio...</td>
      <td>200</td>
      <td>1538717789000</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; r...</td>
      <td>200021</td>
    </tr>
    <tr>
      <th>19</th>
      <td>None</td>
      <td>Logged In</td>
      <td>Liliana</td>
      <td>F</td>
      <td>4</td>
      <td>Owen</td>
      <td>NaN</td>
      <td>free</td>
      <td>Detroit-Warren-Dearborn, MI</td>
      <td>PUT</td>
      <td>Thumbs Up</td>
      <td>1535032914000</td>
      <td>65</td>
      <td>None</td>
      <td>307</td>
      <td>1538717790000</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; r...</td>
      <td>200021</td>
    </tr>
    <tr>
      <th>20</th>
      <td>None</td>
      <td>Logged In</td>
      <td>Liliana</td>
      <td>F</td>
      <td>5</td>
      <td>Owen</td>
      <td>NaN</td>
      <td>free</td>
      <td>Detroit-Warren-Dearborn, MI</td>
      <td>PUT</td>
      <td>Add to Playlist</td>
      <td>1535032914000</td>
      <td>65</td>
      <td>None</td>
      <td>200</td>
      <td>1538717844000</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; r...</td>
      <td>200021</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Be Bop Deluxe</td>
      <td>Logged In</td>
      <td>Liliana</td>
      <td>F</td>
      <td>6</td>
      <td>Owen</td>
      <td>314.59220</td>
      <td>free</td>
      <td>Detroit-Warren-Dearborn, MI</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1535032914000</td>
      <td>65</td>
      <td>Axe Victim</td>
      <td>200</td>
      <td>1538717970000</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; r...</td>
      <td>200021</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Puppetmastaz</td>
      <td>Logged In</td>
      <td>Liliana</td>
      <td>F</td>
      <td>7</td>
      <td>Owen</td>
      <td>276.63628</td>
      <td>free</td>
      <td>Detroit-Warren-Dearborn, MI</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1535032914000</td>
      <td>65</td>
      <td>Stories</td>
      <td>200</td>
      <td>1538718284000</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; r...</td>
      <td>200021</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Tub Ring</td>
      <td>Logged In</td>
      <td>Liliana</td>
      <td>F</td>
      <td>8</td>
      <td>Owen</td>
      <td>233.69098</td>
      <td>free</td>
      <td>Detroit-Warren-Dearborn, MI</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1535032914000</td>
      <td>65</td>
      <td>Invalid</td>
      <td>200</td>
      <td>1538718560000</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; r...</td>
      <td>200021</td>
    </tr>
    <tr>
      <th>24</th>
      <td>None</td>
      <td>Logged In</td>
      <td>Liliana</td>
      <td>F</td>
      <td>9</td>
      <td>Owen</td>
      <td>NaN</td>
      <td>free</td>
      <td>Detroit-Warren-Dearborn, MI</td>
      <td>GET</td>
      <td>Upgrade</td>
      <td>1535032914000</td>
      <td>65</td>
      <td>None</td>
      <td>200</td>
      <td>1538718576000</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; r...</td>
      <td>200021</td>
    </tr>
  </tbody>
</table>
</div>



#### (3) print categorical features with low cardinality (less than 56 distinct values): ####
list = ["auth", "gender", "level", "method", "page", "status", "userAgent"]
value name and value counts


```python
for column in ["auth", "gender", "level", "method", "page", "status", "userAgent"]:
    print(user_log_valid.groupBy(column).count().orderBy("count").show()); 
```

    +---------+------+
    |     auth| count|
    +---------+------+
    |Cancelled|    52|
    |Logged In|278102|
    +---------+------+
    
    None
    +------+------+
    |gender| count|
    +------+------+
    |     M|123576|
    |     F|154578|
    +------+------+
    
    None
    +-----+------+
    |level| count|
    +-----+------+
    | free| 55721|
    | paid|222433|
    +-----+------+
    
    None
    +------+------+
    |method| count|
    +------+------+
    |   GET| 20336|
    |   PUT|257818|
    +------+------+
    
    None
    +--------------------+------+
    |                page| count|
    +--------------------+------+
    |              Cancel|    52|
    |Cancellation Conf...|    52|
    |    Submit Downgrade|    63|
    |      Submit Upgrade|   159|
    |               Error|   252|
    |       Save Settings|   310|
    |               About|   495|
    |             Upgrade|   499|
    |                Help|  1454|
    |            Settings|  1514|
    |           Downgrade|  2055|
    |         Thumbs Down|  2546|
    |              Logout|  3226|
    |         Roll Advert|  3933|
    |          Add Friend|  4277|
    |     Add to Playlist|  6526|
    |                Home| 10082|
    |           Thumbs Up| 12551|
    |            NextSong|228108|
    +--------------------+------+
    
    None
    +------+------+
    |status| count|
    +------+------+
    |   404|   252|
    |   307| 23184|
    |   200|254718|
    +------+------+
    
    None
    +--------------------+-----+
    |           userAgent|count|
    +--------------------+-----+
    |Mozilla/5.0 (X11;...|   62|
    |"Mozilla/5.0 (Mac...|  187|
    |"Mozilla/5.0 (Mac...|  235|
    |"Mozilla/5.0 (Mac...|  240|
    |Mozilla/5.0 (Maci...|  251|
    |"Mozilla/5.0 (Mac...|  379|
    |"Mozilla/5.0 (Win...|  410|
    |"Mozilla/5.0 (Mac...|  512|
    |"Mozilla/5.0 (Mac...|  573|
    |Mozilla/5.0 (comp...|  815|
    |Mozilla/5.0 (Wind...|  972|
    |Mozilla/5.0 (Wind...| 1102|
    |"Mozilla/5.0 (Mac...| 1102|
    |Mozilla/5.0 (comp...| 1245|
    |"Mozilla/5.0 (Mac...| 1262|
    |"Mozilla/5.0 (Mac...| 1322|
    |"Mozilla/5.0 (X11...| 1639|
    |Mozilla/5.0 (X11;...| 1874|
    |Mozilla/5.0 (Maci...| 1950|
    |"Mozilla/5.0 (iPh...| 1976|
    +--------------------+-----+
    only showing top 20 rows
    
    None



```python
# detailled analysis of feature "userAgent"
pd_df = user_log_valid.groupBy("userAgent").count().orderBy("count").toPandas()
print(pd_df["userAgent"].tolist())
pd_df["count"].describe()
```

    ['Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/6.1.5 Safari/537.77.4"', 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)', 'Mozilla/5.0 (Windows NT 6.1; rv:31.0) Gecko/20100101 Firefox/31.0', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.74.9 (KHTML, like Gecko) Version/7.0.2 Safari/537.74.9"', 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.3 (KHTML, like Gecko) Version/8.0 Safari/600.1.3"', '"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_1 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D201 Safari/9537.53"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.76.4 (KHTML, like Gecko) Version/7.0.4 Safari/537.76.4"', '"Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36"', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; rv:31.0) Gecko/20100101 Firefox/31.0', 'Mozilla/5.0 (Windows NT 6.0; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (iPad; CPU OS 7_1_1 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D201 Safari/9537.53"', 'Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D167 Safari/9537.53"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:32.0) Gecko/20100101 Firefox/32.0', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko', '"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.8 (KHTML, like Gecko) Version/8.0 Safari/600.1.8"', 'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"', '"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.78.2 (KHTML, like Gecko) Version/7.0.6 Safari/537.78.2"', 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)', '"Mozilla/5.0 (iPad; CPU OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"']





    count       56.000000
    mean      4967.035714
    std       5602.901751
    min         62.000000
    25%       1257.750000
    50%       2506.500000
    75%       6718.750000
    max      22751.000000
    Name: count, dtype: float64




```python
# further analysis of relationship between page and status...
user_log_valid.groupBy(["status", "method", "page"]).count().orderBy("status", "method", "count").show()
```

    +------+------+--------------------+------+
    |status|method|                page| count|
    +------+------+--------------------+------+
    |   200|   GET|Cancellation Conf...|    52|
    |   200|   GET|               About|   495|
    |   200|   GET|             Upgrade|   499|
    |   200|   GET|                Help|  1454|
    |   200|   GET|            Settings|  1514|
    |   200|   GET|           Downgrade|  2055|
    |   200|   GET|         Roll Advert|  3933|
    |   200|   GET|                Home| 10082|
    |   200|   PUT|     Add to Playlist|  6526|
    |   200|   PUT|            NextSong|228108|
    |   307|   PUT|              Cancel|    52|
    |   307|   PUT|    Submit Downgrade|    63|
    |   307|   PUT|      Submit Upgrade|   159|
    |   307|   PUT|       Save Settings|   310|
    |   307|   PUT|         Thumbs Down|  2546|
    |   307|   PUT|              Logout|  3226|
    |   307|   PUT|          Add Friend|  4277|
    |   307|   PUT|           Thumbs Up| 12551|
    |   404|   GET|               Error|   252|
    +------+------+--------------------+------+
    


#### Findings on (3) ####
* "status": refers to html status codes (https://en.wikipedia.org/wiki/List_of_HTTP_status_codes)
    * 200: page ok
    * 307: Temporary redirect
    * 404: error
* "userAgent": needs to be split up into further features for analysis, e.g. influence of operating system

#### (4) "location": analyse user locations according to Country, State, etc ####


```python
pd_df = user_log_valid.groupBy("location", "userID").count().orderBy("count").toPandas()
pd_df["CSA"] = pd_df["location"].apply(lambda loc_string: loc_string.split(", ")[1])
print("number of unique Combined Statistical Areas in sample data: {}".format(pd_df["CSA"].nunique()))
pd_df.describe()
```

    number of unique Combined Statistical Areas in sample data: 58





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
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>225.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1236.240000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1329.531716</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>296.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>848.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1863.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9632.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# evaluate user dataset row distribution over CSA
pd_CSA_count = pd.pivot_table(pd_df, values = ["userID","count"], index="CSA", aggfunc={
    "userID": "count",
    "count": np.sum})
pd_CSA_count.sort_values(by = ["count", "userID"], ascending = False)
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
      <th>count</th>
      <th>userID</th>
    </tr>
    <tr>
      <th>CSA</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CA</th>
      <td>46771</td>
      <td>33</td>
    </tr>
    <tr>
      <th>NY-NJ-PA</th>
      <td>23684</td>
      <td>15</td>
    </tr>
    <tr>
      <th>TX</th>
      <td>23494</td>
      <td>16</td>
    </tr>
    <tr>
      <th>MA-NH</th>
      <td>13873</td>
      <td>5</td>
    </tr>
    <tr>
      <th>FL</th>
      <td>13190</td>
      <td>14</td>
    </tr>
    <tr>
      <th>NC</th>
      <td>10688</td>
      <td>6</td>
    </tr>
    <tr>
      <th>NC-SC</th>
      <td>7780</td>
      <td>6</td>
    </tr>
    <tr>
      <th>CO</th>
      <td>7493</td>
      <td>4</td>
    </tr>
    <tr>
      <th>MI</th>
      <td>7216</td>
      <td>5</td>
    </tr>
    <tr>
      <th>NJ</th>
      <td>7001</td>
      <td>2</td>
    </tr>
    <tr>
      <th>KY-IN</th>
      <td>6880</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CT</th>
      <td>6720</td>
      <td>7</td>
    </tr>
    <tr>
      <th>PA-NJ-DE-MD</th>
      <td>5890</td>
      <td>5</td>
    </tr>
    <tr>
      <th>IL-IN-WI</th>
      <td>5114</td>
      <td>6</td>
    </tr>
    <tr>
      <th>MO-IL</th>
      <td>4858</td>
      <td>6</td>
    </tr>
    <tr>
      <th>AZ</th>
      <td>4846</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NH</th>
      <td>4764</td>
      <td>2</td>
    </tr>
    <tr>
      <th>VA</th>
      <td>4651</td>
      <td>2</td>
    </tr>
    <tr>
      <th>MS</th>
      <td>4634</td>
      <td>3</td>
    </tr>
    <tr>
      <th>NY</th>
      <td>4536</td>
      <td>5</td>
    </tr>
    <tr>
      <th>GA</th>
      <td>4236</td>
      <td>4</td>
    </tr>
    <tr>
      <th>WA</th>
      <td>3772</td>
      <td>4</td>
    </tr>
    <tr>
      <th>AK</th>
      <td>3563</td>
      <td>2</td>
    </tr>
    <tr>
      <th>KY</th>
      <td>3462</td>
      <td>3</td>
    </tr>
    <tr>
      <th>OH</th>
      <td>3432</td>
      <td>3</td>
    </tr>
    <tr>
      <th>DC-VA-MD-WV</th>
      <td>3090</td>
      <td>4</td>
    </tr>
    <tr>
      <th>PA</th>
      <td>2923</td>
      <td>3</td>
    </tr>
    <tr>
      <th>AL</th>
      <td>2857</td>
      <td>4</td>
    </tr>
    <tr>
      <th>GA-AL</th>
      <td>2716</td>
      <td>2</td>
    </tr>
    <tr>
      <th>MD</th>
      <td>2710</td>
      <td>2</td>
    </tr>
    <tr>
      <th>MO-KS</th>
      <td>2562</td>
      <td>2</td>
    </tr>
    <tr>
      <th>MT</th>
      <td>2386</td>
      <td>2</td>
    </tr>
    <tr>
      <th>WV</th>
      <td>2278</td>
      <td>1</td>
    </tr>
    <tr>
      <th>MN-WI</th>
      <td>2241</td>
      <td>3</td>
    </tr>
    <tr>
      <th>IL</th>
      <td>2102</td>
      <td>3</td>
    </tr>
    <tr>
      <th>NV</th>
      <td>2042</td>
      <td>3</td>
    </tr>
    <tr>
      <th>TN-VA</th>
      <td>1863</td>
      <td>1</td>
    </tr>
    <tr>
      <th>OK</th>
      <td>1842</td>
      <td>1</td>
    </tr>
    <tr>
      <th>TN</th>
      <td>1672</td>
      <td>1</td>
    </tr>
    <tr>
      <th>WI</th>
      <td>1342</td>
      <td>1</td>
    </tr>
    <tr>
      <th>IN</th>
      <td>1262</td>
      <td>3</td>
    </tr>
    <tr>
      <th>LA</th>
      <td>1171</td>
      <td>2</td>
    </tr>
    <tr>
      <th>UT</th>
      <td>1102</td>
      <td>1</td>
    </tr>
    <tr>
      <th>IL-MO</th>
      <td>1003</td>
      <td>1</td>
    </tr>
    <tr>
      <th>MD-WV</th>
      <td>982</td>
      <td>1</td>
    </tr>
    <tr>
      <th>RI-MA</th>
      <td>927</td>
      <td>2</td>
    </tr>
    <tr>
      <th>SC-NC</th>
      <td>837</td>
      <td>1</td>
    </tr>
    <tr>
      <th>PA-NJ</th>
      <td>815</td>
      <td>1</td>
    </tr>
    <tr>
      <th>SC</th>
      <td>668</td>
      <td>2</td>
    </tr>
    <tr>
      <th>IA</th>
      <td>651</td>
      <td>1</td>
    </tr>
    <tr>
      <th>AR</th>
      <td>520</td>
      <td>1</td>
    </tr>
    <tr>
      <th>UT-ID</th>
      <td>317</td>
      <td>1</td>
    </tr>
    <tr>
      <th>VA-NC</th>
      <td>246</td>
      <td>2</td>
    </tr>
    <tr>
      <th>NE-IA</th>
      <td>187</td>
      <td>1</td>
    </tr>
    <tr>
      <th>OR-WA</th>
      <td>119</td>
      <td>2</td>
    </tr>
    <tr>
      <th>OH-KY-IN</th>
      <td>88</td>
      <td>2</td>
    </tr>
    <tr>
      <th>TN-MS-AR</th>
      <td>62</td>
      <td>1</td>
    </tr>
    <tr>
      <th>OR</th>
      <td>23</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



####  (5) convert to time/date: ####
list = ["registration", "ts"]


```python
def convert_ts_to_datetime(df, column_list):
    get_datetime = udf(lambda timestamp: datetime.datetime.fromtimestamp(timestamp/1000).isoformat())
    for column in column_list:
        df = df.withColumn(column + "_timestamp", get_datetime(df[column]).cast(TimestampType()))
    return df
```


```python
column_list = ["registration", "ts"]
user_log_valid = convert_ts_to_datetime(user_log_valid, column_list)
```


```python
user_log_valid.printSchema()
```

    root
     |-- artist: string (nullable = true)
     |-- auth: string (nullable = true)
     |-- firstName: string (nullable = true)
     |-- gender: string (nullable = true)
     |-- itemInSession: long (nullable = true)
     |-- lastName: string (nullable = true)
     |-- length: double (nullable = true)
     |-- level: string (nullable = true)
     |-- location: string (nullable = true)
     |-- method: string (nullable = true)
     |-- page: string (nullable = true)
     |-- registration: long (nullable = true)
     |-- sessionId: long (nullable = true)
     |-- song: string (nullable = true)
     |-- status: long (nullable = true)
     |-- ts: long (nullable = true)
     |-- userAgent: string (nullable = true)
     |-- userId: string (nullable = true)
     |-- registration_timestamp: timestamp (nullable = true)
     |-- ts_timestamp: timestamp (nullable = true)
    


#### (6.1) "ts": analyse spread of date/ time


```python
min_date, max_date = user_log_valid.select(min("ts_timestamp"), max("ts_timestamp")).first()
min_date, max_date
```




    (datetime.datetime(2018, 10, 1, 0, 1, 57),
     datetime.datetime(2018, 12, 3, 1, 11, 16))




```python
print("Analyze log data over time:")
pd_df = user_log_valid.select(hour("ts_timestamp").alias("hour")).groupBy("hour").count().orderBy("hour").toPandas()
pd_df.plot.line(x="hour", y="count");
```

    Analyze log data over time:



![png](output_47_1.png)


#### (6.1) "registration": analyse spread of date/ time


```python
pd_df = user_log_valid.select(to_date("registration_timestamp").alias("registration_date")).groupBy("registration_date").count().orderBy("registration_date").toPandas()
pd_df.plot.line(x="registration_date", y="count");
```


![png](output_49_0.png)


#### "ts": further conversion to features for year/ month/ time


```python
user_log_valid = user_log_valid.withColumn("ts_hour", hour("ts_timestamp"))
```

## Features from "page" value ##


```python
print("page values with user depending on method:")
user_log_valid.groupBy("method", "page").count().orderBy("method", "count").show()   
```

    page values with user depending on method:
    +------+--------------------+------+
    |method|                page| count|
    +------+--------------------+------+
    |   GET|Cancellation Conf...|    52|
    |   GET|               Error|   252|
    |   GET|               About|   495|
    |   GET|             Upgrade|   499|
    |   GET|                Help|  1454|
    |   GET|            Settings|  1514|
    |   GET|           Downgrade|  2055|
    |   GET|         Roll Advert|  3933|
    |   GET|                Home| 10082|
    |   PUT|              Cancel|    52|
    |   PUT|    Submit Downgrade|    63|
    |   PUT|      Submit Upgrade|   159|
    |   PUT|       Save Settings|   310|
    |   PUT|         Thumbs Down|  2546|
    |   PUT|              Logout|  3226|
    |   PUT|          Add Friend|  4277|
    |   PUT|     Add to Playlist|  6526|
    |   PUT|           Thumbs Up| 12551|
    |   PUT|            NextSong|228108|
    +------+--------------------+------+
    


### Selection of page values for new features:
* "churn" from "Cancel"
* "downgraded" from "Submit Downgrad"
* ...


```python
def create_page_value_feature(df, page_value, col_name):
    '''
    ARGS:
    OUTPUT
    
    Function that creates a new feature from a certain value of feature "page"
    '''
    flag_page_value_event = udf(lambda page: 1 if page == page_value else 0, IntegerType())
    return df.withColumn(col_name, flag_page_value_event("page"))
```


```python
# testing only
test_df = user_log_valid
```


```python
page_value_feature_dict = {"Submit Downgrade" : "downgraded",
                           "Cancel" : "churn",
                           "Submit Upgrade" : "upgraded",
                           "Roll Advert" : "advert_shown",
                           "Thumbs Down": "thumps_down",
                           "Thumbs Up": "thumps_up",
                           "Add Friend": "friend_added",
                           "Add to Playlist" : "song_added"
                          }

for page_value in page_value_feature_dict.keys():
    column_name = page_value_feature_dict[page_value]
    test_df =  create_page_value_feature(test_df, page_value, column_name)
```


```python
test_df.printSchema()
```

    root
     |-- artist: string (nullable = true)
     |-- auth: string (nullable = true)
     |-- firstName: string (nullable = true)
     |-- gender: string (nullable = true)
     |-- itemInSession: long (nullable = true)
     |-- lastName: string (nullable = true)
     |-- length: double (nullable = true)
     |-- level: string (nullable = true)
     |-- location: string (nullable = true)
     |-- method: string (nullable = true)
     |-- page: string (nullable = true)
     |-- registration: long (nullable = true)
     |-- sessionId: long (nullable = true)
     |-- song: string (nullable = true)
     |-- status: long (nullable = true)
     |-- ts: long (nullable = true)
     |-- userAgent: string (nullable = true)
     |-- userId: string (nullable = true)
     |-- registration_timestamp: timestamp (nullable = true)
     |-- ts_timestamp: timestamp (nullable = true)
     |-- ts_hour: integer (nullable = true)
     |-- downgraded: integer (nullable = true)
     |-- churn: integer (nullable = true)
     |-- upgraded: integer (nullable = true)
     |-- advert_shown: integer (nullable = true)
     |-- thumps_down: integer (nullable = true)
     |-- thumps_up: integer (nullable = true)
     |-- friend_added: integer (nullable = true)
     |-- song_added: integer (nullable = true)
    


### Descriptive analysis on features "downgraded", "upgraded" and "churn"


```python
subset_list = list(page_value_feature_dict.values())
subset_list.append("userId")

test_df.select(subset_list).filter(
    (test_df["churn"]==1) | 
    (test_df["downgraded"]==1) | 
    (test_df["advert_shown"]==1) | 
    (test_df["thumps_down"]==1) |
    (test_df["thumps_up"]==1) | 
    (test_df["friend_added"]==1) |
    (test_df["song_added"]==1) |
    (test_df["upgraded"]==1)).count()
```




    30107




```python
print("Descriptive Analysis of userId's that churned, downgraded or upgraded")
pd_df = test_df.select(subset_list).filter(
    (test_df["churn"]==1) | 
    (test_df["downgraded"]==1) | 
    (test_df["advert_shown"]==1) | 
    (test_df["thumps_down"]==1) |
    (test_df["thumps_up"]==1) | 
    (test_df["friend_added"]==1) |
    (test_df["song_added"]==1) |
    (test_df["upgraded"]==1)).toPandas()
```

    Descriptive Analysis of userId's that churned, downgraded or upgraded



```python
def check_churn(userId):
    return True if (pd_df["churn"][pd_df["userId"]==userId].max() >=1) else False

pd_df["user_churned"] = pd_df["userId"].apply(lambda user: check_churn(user))  
```


```python
print("value count of page value features of users depending on feature churn:")
value_count = pd_df.pivot_table(index="user_churned", values=subset_list, aggfunc={
    "userId": pd.Series.nunique,
    "churn" : np.sum,
    "downgraded": np.sum,
    "upgraded": np.sum,
    'advert_shown': np.sum,
    'thumps_down': np.sum,
    'thumps_up': np.sum,
    'friend_added': np.sum,
    'song_added': np.sum
})
value_count.head()
```

    value count of page value features of users depending on feature churn:





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
      <th>advert_shown</th>
      <th>churn</th>
      <th>downgraded</th>
      <th>friend_added</th>
      <th>song_added</th>
      <th>thumps_down</th>
      <th>thumps_up</th>
      <th>upgraded</th>
      <th>userId</th>
    </tr>
    <tr>
      <th>user_churned</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>2966</td>
      <td>0</td>
      <td>54</td>
      <td>3641</td>
      <td>5488</td>
      <td>2050</td>
      <td>10692</td>
      <td>127</td>
      <td>172</td>
    </tr>
    <tr>
      <th>True</th>
      <td>967</td>
      <td>52</td>
      <td>9</td>
      <td>636</td>
      <td>1038</td>
      <td>496</td>
      <td>1859</td>
      <td>32</td>
      <td>52</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_log = test_df
```

## Ideas for other features
* per user: time from registration to last log action
* existing categorical features with low number of distinct values
* user interaction over time (number of songs...)
* tbd

# Feature Engineering
Once you've familiarized yourself with the data, build out the features you find promising to train your model on. To work with the full dataset, you can follow the following steps.
- Write a script to extract the necessary features from the smaller subset of data
- Ensure that your script is scalable, using the best practices discussed in Lesson 3
- Try your script on the full data set, debugging your script if necessary

If you are working in the classroom workspace, you can just extract features based on the small subset of data contained here. Be sure to transfer over this work to the larger dataset when you work on your Spark cluster.

## Create new dataframe for features bases on userId's


```python
# create new df for features
all_users_collect = user_log.select("userId").distinct().collect()
all_users = set([int(row["userId"]) for row in all_users_collect])
features_df = spark.createDataFrame(all_users, IntegerType()).withColumnRenamed('value', 'userId')
```


```python
features_df
```




    DataFrame[value: int]




```python

```


```python

```

# Modeling
Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. Evaluate the accuracy of the various models, tuning parameters as necessary. Determine your winning model based on test accuracy and report results on the validation set. Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.


```python
test_df.printSchema()
```

    root
     |-- artist: string (nullable = true)
     |-- auth: string (nullable = true)
     |-- firstName: string (nullable = true)
     |-- gender: string (nullable = true)
     |-- itemInSession: long (nullable = true)
     |-- lastName: string (nullable = true)
     |-- length: double (nullable = true)
     |-- level: string (nullable = true)
     |-- location: string (nullable = true)
     |-- method: string (nullable = true)
     |-- page: string (nullable = true)
     |-- registration: long (nullable = true)
     |-- sessionId: long (nullable = true)
     |-- song: string (nullable = true)
     |-- status: long (nullable = true)
     |-- ts: long (nullable = true)
     |-- userAgent: string (nullable = true)
     |-- userId: string (nullable = true)
     |-- registration_timestamp: timestamp (nullable = true)
     |-- ts_timestamp: timestamp (nullable = true)
     |-- ts_hour: integer (nullable = true)
     |-- downgraded: integer (nullable = true)
     |-- churn: integer (nullable = true)
     |-- upgraded: integer (nullable = true)
     |-- advert_shown: integer (nullable = true)
     |-- thumps_down: integer (nullable = true)
     |-- thumps_up: integer (nullable = true)
     |-- friend_added: integer (nullable = true)
     |-- song_added: integer (nullable = true)
    



```python
model_df = test_df.select("")
```

## Split in training, test, validation set


```python
#user_log_valid.persist()
training, test, validation = user_log_valid.randomSplit([0.8, 0.1, 0.1], seed=42)
```

## Build Pipeline


```python
# build pipeline

lr = LogisticRegression(maxIter=10, regParam=0.0, elasticNetParam=0)

pipeline = Pipeline(stages=[lr])
```

## Tune Model


```python
# tune model
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam,[0.0, 0.1]) \
    .build()


crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=3)
```


```python
cvModel_q1 = crossval.fit(training)
```


```python
cvModel_q1.avgMetrics
```


```python
results = cvModel_q1.transform(test)
```

## Compute Accuracy of Best Model


```python
# TODO: change label or create feature label

correct_results = (results.filter(results.label == results.prediction).count())
total_results= (results.count())
accuracy = correct_results/ total_results
```

# Final Steps
Clean up your code, adding comments and renaming variables to make the code easier to read and maintain. Refer to the Spark Project Overview page and Data Scientist Capstone Project Rubric to make sure you are including all components of the capstone project and meet all expectations. Remember, this includes thorough documentation in a README file in a Github repository, as well as a web app or blog post.


```python

```
