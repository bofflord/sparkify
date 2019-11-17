
# Sparkify Project Workspace
This workspace contains a tiny subset (128MB) of the full dataset available (12GB). Feel free to use this workspace to build your project, or to explore a smaller subset with Spark before deploying your cluster on the cloud. Instructions for setting up your Spark cluster is included in the last lesson of the Extracurricular Spark Course content.

You can follow the steps below to guide your data analysis and model building portion of this project.


```
# import libraries

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, Window

#from pyspark.sql.functions import udf, sum as Fsum, desc, asc, countDistinct, col, to_date, year, month, dayofmonth, minute, hour, datediff, min, max, isnull, when
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


```
# create a Spark session
spark = SparkSession \
    .builder \
    .appName("Sparkify") \
    .getOrCreate()
```


```
# Install additional libraries via pip in the current Jupyter kernel
import sys
!{sys.executable} -m pip install pixiedust
```

    Collecting pixiedust
    [?25l  Downloading https://files.pythonhosted.org/packages/bc/a8/e84b2ed12ee387589c099734b6f914a520e1fef2733c955982623080e813/pixiedust-1.1.17.tar.gz (197kB)
    [K    100% |████████████████████████████████| 204kB 2.2MB/s ta 0:00:01
    [?25hCollecting mpld3 (from pixiedust)
    [?25l  Downloading https://files.pythonhosted.org/packages/91/95/a52d3a83d0a29ba0d6898f6727e9858fe7a43f6c2ce81a5fe7e05f0f4912/mpld3-0.3.tar.gz (788kB)
    [K    100% |████████████████████████████████| 798kB 3.4MB/s ta 0:00:01
    [?25hRequirement already satisfied: lxml in /opt/conda/lib/python3.6/site-packages (from pixiedust) (4.1.1)
    Collecting geojson (from pixiedust)
      Downloading https://files.pythonhosted.org/packages/e4/8d/9e28e9af95739e6d2d2f8d4bef0b3432da40b7c3588fbad4298c1be09e48/geojson-2.5.0-py2.py3-none-any.whl
    Collecting astunparse (from pixiedust)
      Downloading https://files.pythonhosted.org/packages/2e/37/5dd0dd89b87bb5f0f32a7e775458412c52d78f230ab8d0c65df6aabc4479/astunparse-1.6.2-py2.py3-none-any.whl
    Requirement already satisfied: markdown in /opt/conda/lib/python3.6/site-packages (from pixiedust) (2.6.9)
    Requirement already satisfied: colour in /opt/conda/lib/python3.6/site-packages (from pixiedust) (0.1.5)
    Requirement already satisfied: requests in /opt/conda/lib/python3.6/site-packages (from pixiedust) (2.18.4)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /opt/conda/lib/python3.6/site-packages (from astunparse->pixiedust) (0.30.0)
    Requirement already satisfied: six<2.0,>=1.6.1 in /opt/conda/lib/python3.6/site-packages (from astunparse->pixiedust) (1.11.0)
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



```
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
    Table SPARK_PACKAGES created successfully
    Table USER_PREFERENCES created successfully
    Table service_connections created successfully



```
pixiedust.optOut()
```

    Pixiedust will not collect anonymous install statistics.


# Load and Clean Dataset
In this workspace, the mini-dataset file is `mini_sparkify_event_data.json`. Load and clean the dataset, checking for invalid or missing data - for example, records without userids or sessionids. 


```
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


```
display(user_log)
```


```
for column in user_log.columns:
    print("Analysis of column {}".format(column))
    print("Statistical properties:")
    print(user_log.describe(column).show())
    print("\nCount of unique values in column:")
    print(user_log.select(F.countDistinct(column)).show(),"\n")
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
    



```
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
    



```
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

#### (4) "location": analyse user locations and CSA (Combined Statistical Areas)


```
# create new feature CSA (Combined Statistical Areas) from location 
get_csa = F.split(user_log["location"], ", ")
user_log = user_log.withColumn("CSA", get_csa.getItem(1))
```


```
distinct_users_csa = user_log.select("userId", "CSA").distinct().count()
distinct_users_location = user_log.select("userId", "location").distinct().count()
print("Distinct users per CSA: {}. Distinct users per location: {}".format(distinct_users_csa, distinct_users_location))
```

    Distinct users per CSA: 226. Distinct users per location: 226


####  (5) convert to time/date: ####
list = ["registration", "ts"]


```
def convert_ts_to_datetime(df, column):
    get_datetime = F.udf(lambda timestamp: datetime.datetime.fromtimestamp(timestamp/1000).isoformat())
    df = df.withColumn(column + "_ts", get_datetime(df[column]).cast(TimestampType()))
    return df
```


```
# create new features in timestamp format from features "registration", "ts"
#column_list = ["registration", "ts"]
user_log = convert_ts_to_datetime(user_log, "ts")
```

#### (6) "ts": analyse spread of date/ time


```
min_date, max_date = user_log.select(F.min("ts_ts"), F.max("ts_ts")).first()
print("Minimum and Maximum timestamp data:")
min_date, max_date
```

    Minimum and Maximum timestamp data:





    (datetime.datetime(2018, 10, 1, 0, 1, 57),
     datetime.datetime(2018, 12, 3, 1, 11, 16))



#### (7) "ts": further conversion to features for date/ hour


```
# get new features day and hour
user_log = user_log.withColumn("ts_hour", F.hour("ts_ts"))
user_log = user_log.withColumn("ts_date", F.to_date("ts_ts"))
```


```
print("Analyze log data over time:")
#pd_df = user_log.select(hour("ts_ts").alias("hour")).groupBy("hour").count().orderBy("hour").toPandas()
pd_df = user_log.select("ts_hour").groupBy("ts_hour").count().orderBy("ts_hour").toPandas()
pd_df.plot.line(x="ts_hour", y="count");
```

    Analyze log data over time:



![png](output_28_1.png)


## Features from "page" value ##

### Selection of page values for new features:
* "downgraded" from "Submit Downgrad"
* ...


```
def create_page_value_feature(df, page_value, col_name):
    '''
    ARGS:
    OUTPUT
    
    Function that creates a new feature from a certain value of feature "page"
    '''
    flag_page_value_event = F.udf(lambda page: 1 if page == page_value else 0, IntegerType())
    return df.withColumn(col_name, flag_page_value_event("page"))
```


```
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

## Define feature "churn"


```
flag_churn_event = F.udf(lambda page: 1 if page == "Cancel" else 0, IntegerType())
user_log = user_log.withColumn("churn", flag_churn_event("page"))
```

# Feature Engineering
Once you've familiarized yourself with the data, build out the features you find promising to train your model on. To work with the full dataset, you can follow the following steps.
- Write a script to extract the necessary features from the smaller subset of data
- Ensure that your script is scalable, using the best practices discussed in Lesson 3
- Try your script on the full data set, debugging your script if necessary

If you are working in the classroom workspace, you can just extract features based on the small subset of data contained here. Be sure to transfer over this work to the larger dataset when you work on your Spark cluster.

## Explore data with regards to churn


```
all_users_count = user_log.select("userId").filter(user_log["userId"]!="").distinct().count()
churned_users_count = user_log.select("userId").filter(user_log["churn"]==1).distinct().count()
print("Out of a total {} users, {} users churned. These are {:.1f} %".format(
all_users_count, churned_users_count, churned_users_count*100/all_users_count))
```

    Out of a total 225 users, 52 users churned. These are 23.1 %


## Create new dataframe for features bases on userId's


```
# create new df for features
all_users_collect = user_log.select("userId").filter(user_log["userId"]!="").distinct().collect()
all_users = set([int(row["userId"]) for row in all_users_collect])
features_df = spark.createDataFrame(all_users, IntegerType()).withColumnRenamed('value', 'userId')
```

## Encode label "churned users"


```
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


```
# one hot encode gender in original df
one_hot_encode_gender = F.udf(lambda gender: 1 if gender == "M" else 0, IntegerType())
user_log = user_log.withColumn("gender_bin", one_hot_encode_gender("gender"))
```


```
# join binary gender on userId in features df
user_gender_selection =  user_log.select(["userId", "gender_bin"]).dropDuplicates(subset=['userId'])
features_df = features_df.join(user_gender_selection, "userId")
```

### Encode "level"
* level value "paid" = value 1
* level value "free" = value 0


```
# one hot encode level in original df
one_hot_encode_level = F.udf(lambda level: 1 if level == "paid" else 0, IntegerType())
user_log = user_log.withColumn("level_bin", one_hot_encode_level("level"))
```


```
# join binary gender on userId in features df
user_level_selection =  user_log.select(["userId", "level_bin"]).dropDuplicates(subset=['userId'])
features_df = features_df.join(user_level_selection, "userId")
```

## Encode page view features
* encode count of page view features per userId
* page view features to be included: 
['downgraded',
 'upgraded',
 'advert_shown',
 'thumps_down',
 'thumps_up',
 'friend_added',
 'song_added']


```
page_features_count = user_log.groupBy("userId").sum('downgraded', 'upgraded',
 'advert_shown',
 'thumps_down',
 'thumps_up',
 'friend_added',
 'song_added')
features_df = features_df.join(page_features_count, "userId", how="left")
```

## Encode further features - tbd +++++++++++++
* "song_count": songs per user
* "days_since_reg": days from registration until latest user timestamp of a user


```
user_log.schema.names
```




    ['artist',
     'auth',
     'firstName',
     'gender',
     'itemInSession',
     'lastName',
     'length',
     'level',
     'location',
     'method',
     'page',
     'registration',
     'sessionId',
     'song',
     'status',
     'ts',
     'userAgent',
     'userId',
     'CSA',
     'ts_ts',
     'ts_hour',
     'ts_date',
     'downgraded',
     'upgraded',
     'advert_shown',
     'thumps_down',
     'thumps_up',
     'friend_added',
     'song_added',
     'churn',
     'gender_bin',
     'level_bin']




```
# create new feature "song_count" in features_df
song_count = user_log.groupBy("userId").agg(F.count("song").alias("song_count")).orderBy("song_count", ascending=False)
features_df = features_df.join(song_count, "userId", how="left")
```


```
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


```
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

# Feature selection


```
display(features_df)
```


<style type="text/css">.pd_warning{display:none;}</style><div class="pd_warning"><em>Hey, there's something awesome here! To see it, open this notebook outside GitHub, in a viewer like Jupyter</em></div>



```

```

## Feature scaling and vectorization


```
features_df.schema.names
```




    ['userId',
     'label',
     'gender_bin',
     'level_bin',
     'sum(downgraded)',
     'sum(upgraded)',
     'sum(advert_shown)',
     'sum(thumps_down)',
     'sum(thumps_up)',
     'sum(friend_added)',
     'sum(song_added)',
     'sum(downgraded)_scaled',
     'sum(upgraded)_scaled',
     'sum(advert_shown)_scaled',
     'sum(thumps_down)_scaled',
     'sum(thumps_up)_scaled',
     'sum(friend_added)_scaled',
     'sum(song_added)_scaled',
     'features',
     'song_count',
     'days_since_reg',
     'total_session_time_sec']



### Vectorize and scale non-binary features
* Vectorization via VectorAssembler
* Scaling via MinMaxScaler
* user Pipeline to combine both in the transformation process


```
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


```
convert_vector_to_double = F.udf(lambda vector_value: round(float(list(vector_value)[0]),3), DoubleType())


for column in nonbinary_feature_list:
    # convert column to vector via VectorAssembler
    assembler = VectorAssembler(inputCols=[column], outputCol=column+"_vect")
    # Scale vectorized column
    scaler = MinMaxScaler(inputCol=column+"_vect", outputCol=column+"_scaled")
    # create Pipeline with assembler and scaler
    pipeline = Pipeline(stages=[assembler, scaler])
    # apply pipelien on features_df Dataframe
    features_df = pipeline.fit(features_df).transform(features_df) \
    .withColumn(column+"_scaled", convert_vector_to_double(column+"_scaled")).drop(column+"_vect")
```


```
features_df.printSchema()
```

### Merge scaled features to one feature vector


```
# create feature list that shall be merged in on vector
feature_list = features_df.schema.names
# remove columns userId, label and all items in nonbinary_feature_list
remove_features_list= nonbinary_feature_list + ["userId", "label"]
feature_list = [item for item in feature_list if item not in remove_features_list]
# assemble features in feature_list to one vector using VectorAssembler
assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
features_df = assembler.transform(features_df)
```


```
features_df.printSchema()
```

## Perform PCA to select relevant features


```
pca = PCA(k=5, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(features_df["features"])

result = model.transform(features_df["features"]).select("pcaFeatures")
result.show(truncate=False)
```

# Modeling
Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. Evaluate the accuracy of the various models, tuning parameters as necessary. Determine your winning model based on test accuracy and report results on the validation set. Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.

## Split in training, test, validation set


```
train, test = features_df.randomSplit([0.8, 0.2], seed=42)

plt.hist(features_df.toPandas()['label'])
plt.show()
```


![png](output_72_0.png)


### Analyze label class imbalance - tbd +++++++++++++


```
# calculate balancing ratio for account for class imbalance

balancing_ratio = train.filter(train['label']==0).count()/train.count()
train=train.withColumn("classWeights", F.when(train.label == 1,balancing_ratio).otherwise(1-balancing_ratio))
```

## Machine Learning Model Selection, Tuning and Evaluation
 * Model learning problem category: supervised learning, logistic regression
 * ML estimators from pyspark.ml:
     * LogisticRegression
     * tbd
 * ML hyperparameters in estimators (for grid search/ tuning):
     * LogisticRegression(maxIter=10, regParam=0.0, elasticNetParam=0)
     * tbd
 * ML evaluators from pyspark.ml:
     * BinaryClassificationEvaluator
     * tbd


```
# Create a logistic regression object
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', weightCol="classWeights")

# create evaluator
evaluator = BinaryClassificationEvaluator(metricName = 'areaUnderPR')

```


```
# create lr_model
lr_model = lr.fit(train)
training_summary = lr_model.summary
```


```
# ToDo: evaluate training summary ++++++++++++++++

# TBD
```

## Tune Model
* use cross validation via CrossValidator and paramGrid


```
# build paramGrid and cross validator
paramGrid = (ParamGridBuilder() \
    .addGrid(lr.maxIter, [1, 5, 10]) \
    .addGrid(lr.regParam,[0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build())


crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)
```


```
# run cross validation
# train model on train data
crossval_model = crossval.fit(train)
# predict on test data
pred = crossval_model.transform(test)
```


```
#cvModel_q1 = crossval.fit(training)
```


```
#cvModel_q1.avgMetrics
```


```
#results = cvModel_q1.transform(test)
```

## Evaluate results
* use scikit learn metrics f1, precision, recall for model evaluation


```
# evaluate results
pd_pred = pred.toPandas()
```


```
pd_pred.head()
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
      <th>userId</th>
      <th>label</th>
      <th>gender_bin</th>
      <th>level_bin</th>
      <th>sum(downgraded)</th>
      <th>sum(upgraded)</th>
      <th>sum(advert_shown)</th>
      <th>sum(thumps_down)</th>
      <th>sum(thumps_up)</th>
      <th>sum(friend_added)</th>
      <th>...</th>
      <th>sum(upgraded)_scaled</th>
      <th>sum(advert_shown)_scaled</th>
      <th>sum(thumps_down)_scaled</th>
      <th>sum(thumps_up)_scaled</th>
      <th>sum(friend_added)_scaled</th>
      <th>sum(song_added)_scaled</th>
      <th>features</th>
      <th>rawPrediction</th>
      <th>probability</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>137</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>8</td>
      <td>8</td>
      <td>...</td>
      <td>0.25</td>
      <td>0.078</td>
      <td>0.013</td>
      <td>0.018</td>
      <td>0.056</td>
      <td>0.017</td>
      <td>[1.0, 0.0, 0.0, 0.25, 0.078, 0.013, 0.018, 0.0...</td>
      <td>[-0.582222225032, 0.582222225032]</td>
      <td>[0.358421420169, 0.641578579831]</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>17</td>
      <td>111</td>
      <td>53</td>
      <td>...</td>
      <td>0.25</td>
      <td>0.062</td>
      <td>0.227</td>
      <td>0.254</td>
      <td>0.371</td>
      <td>0.283</td>
      <td>[1.0, 0.0, 0.0, 0.25, 0.062, 0.227, 0.254, 0.3...</td>
      <td>[0.896119455314, -0.896119455314]</td>
      <td>[0.710151397173, 0.289848602827]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>133</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.008</td>
      <td>0.000</td>
      <td>0.005</td>
      <td>0.007</td>
      <td>0.013</td>
      <td>[1.0, 0.0, 0.0, 0.0, 0.008, 0.0, 0.005, 0.007,...</td>
      <td>[-0.48276677666, 0.48276677666]</td>
      <td>[0.381599004404, 0.618400995596]</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>78</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>3</td>
      <td>11</td>
      <td>2</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.125</td>
      <td>0.040</td>
      <td>0.025</td>
      <td>0.014</td>
      <td>0.037</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.125, 0.04, 0.025, 0.014...</td>
      <td>[-0.261030769771, 0.261030769771]</td>
      <td>[0.435110339059, 0.564889660941]</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>76</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>2</td>
      <td>13</td>
      <td>3</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.117</td>
      <td>0.027</td>
      <td>0.030</td>
      <td>0.021</td>
      <td>0.017</td>
      <td>[1.0, 0.0, 0.0, 0.0, 0.117, 0.027, 0.03, 0.021...</td>
      <td>[-0.569896815398, 0.569896815398]</td>
      <td>[0.361260634501, 0.638739365499]</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```
# calculate score for f1, precision, recall
f1 = f1_score(pd_pred.label, pd_pred.prediction)
recall = recall_score(pd_pred.label, pd_pred.prediction)
precision = precision_score(pd_pred.label, pd_pred.prediction)
```


```
print("F1 Score: {:.2f}, Recall: {:.2f}, Precision: {:.2f}".format(f1, recall, precision))
```

    F1 Score: 0.29, Recall: 0.40, Precision: 0.22



```
# TODO: change label or create feature label

#correct_results = (results.filter(results.label == results.prediction).count())
#total_results= (results.count())
#accuracy = correct_results/ total_results
```

# Final Steps
Clean up your code, adding comments and renaming variables to make the code easier to read and maintain. Refer to the Spark Project Overview page and Data Scientist Capstone Project Rubric to make sure you are including all components of the capstone project and meet all expectations. Remember, this includes thorough documentation in a README file in a Github repository, as well as a web app or blog post.


```

```
