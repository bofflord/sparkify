
# Sparkify Project Workspace


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
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier

# for model evaluation
from sklearn.metrics import f1_score, recall_score, precision_score

# for Principal Component Analysis
from pyspark.ml.feature import PCA

# import IBM Watson Studio Lib to save data and models
from project_lib import Project
project = Project(sc,"0f767c58-a6da-4225-a3a9-1526012f97c0", "p-3cdd7b2b8037c9f4b97413e96fdcb72741dcfb9d")

# import time to pause program execution during model training
import time 
```

    Waiting for a Spark session to start...
    Spark Initialization Done! ApplicationId = app-20191229142350-0001
    KERNEL_ID = 5f9bea9b-baf6-4ae9-919b-b14ca1d17425
    

# Start SparkSession and load dataset


```python

import ibmos2spark
# @hidden_cell
credentials = {
    'endpoint': 'https://s3.eu-geo.objectstorage.service.networklayer.com',
    'service_id': 'iam-ServiceId-f92cad2e-dfc4-460b-b6da-6823a8a1941c',
    'iam_service_endpoint': 'https://iam.eu-gb.bluemix.net/oidc/token',
    'api_key': 'Z6Nf1UKJ_bt7cKZvw_EiAHYIoYhtcCpwOmkJJT3Slxy_'
}

configuration_name = 'os_1647a8daaebd47249daa2481cc9164f0_configs'
cos = ibmos2spark.CloudObjectStorage(sc, credentials, configuration_name, 'bluemix_cos')

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
# Since JSON data can be semi-structured and contain additional metadata, it is possible that you might face issues with the DataFrame layout.
# Please read the documentation of 'SparkSession.read()' to learn more about the possibilities to adjust the data loading.
# PySpark documentation: http://spark.apache.org/docs/2.0.2/api/python/pyspark.sql.html#pyspark.sql.DataFrameReader.json

user_log = spark.read.json(cos.url('medium-sparkify-event-data.json', 'sparkify-donotdelete-pr-2exnp1jnopynlt'))
user_log.take(5)

```




    [Row(artist='Martin Orford', auth='Logged In', firstName='Joseph', gender='M', itemInSession=20, lastName='Morales', length=597.55057, level='free', location='Corpus Christi, TX', method='PUT', page='NextSong', registration=1532063507000, sessionId=292, song='Grand Designs', status=200, ts=1538352011000, userAgent='"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', userId='293'),
     Row(artist="John Brown's Body", auth='Logged In', firstName='Sawyer', gender='M', itemInSession=74, lastName='Larson', length=380.21179, level='free', location='Houston-The Woodlands-Sugar Land, TX', method='PUT', page='NextSong', registration=1538069638000, sessionId=97, song='Bulls', status=200, ts=1538352025000, userAgent='"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', userId='98'),
     Row(artist='Afroman', auth='Logged In', firstName='Maverick', gender='M', itemInSession=184, lastName='Santiago', length=202.37016, level='paid', location='Orlando-Kissimmee-Sanford, FL', method='PUT', page='NextSong', registration=1535953455000, sessionId=178, song='Because I Got High', status=200, ts=1538352118000, userAgent='"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', userId='179'),
     Row(artist=None, auth='Logged In', firstName='Maverick', gender='M', itemInSession=185, lastName='Santiago', length=None, level='paid', location='Orlando-Kissimmee-Sanford, FL', method='PUT', page='Logout', registration=1535953455000, sessionId=178, song=None, status=307, ts=1538352119000, userAgent='"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', userId='179'),
     Row(artist='Lily Allen', auth='Logged In', firstName='Gianna', gender='F', itemInSession=22, lastName='Campos', length=194.53342, level='paid', location='Mobile, AL', method='PUT', page='NextSong', registration=1535931018000, sessionId=245, song='Smile (Radio Edit)', status=200, ts=1538352124000, userAgent='Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', userId='246')]




```python
import pixiedust
```

    Pixiedust database opened successfully
    



        <div style="margin:10px">
            <a href="https://github.com/ibm-watson-data-lab/pixiedust" target="_new">
                <img src="https://github.com/ibm-watson-data-lab/pixiedust/raw/master/docs/_static/pd_icon32.png" style="float:left;margin-right:10px"/>
            </a>
            <span>Pixiedust version 1.1.16</span>
        </div>
        



<div>Warning: You are not running the latest version of PixieDust. Current is 1.1.16, Latest is 1.1.18</div>




                <div>Please copy and run the following command in a new cell to upgrade: <span style="background-color:#ececec;font-family:monospace;padding:0 5px">!pip install --user --upgrade pixiedust</span></div>
            



<div>Please restart kernel after upgrading.</div>



```python
pixiedust.optOut()
```

    Pixiedust will not collect anonymous install statistics.
    


```python
start_time = datetime.datetime.now()
```

# Explore and clean data set


```python
# Peek at dataset
display(user_log)
```


<style type="text/css">.pd_warning{display:none;}</style><div class="pd_warning"><em>Hey, there's something awesome here! To see it, open this notebook outside GitHub, in a viewer like Jupyter</em></div>
        <div class="pd_save is-viewer-good" style="padding-right:10px;text-align: center;line-height:initial !important;font-size: xx-large;font-weight: 500;color: coral;">
            
        </div>
    <div id="chartFigurec58ef392" class="pd_save is-viewer-good" style="overflow-x:auto">
            <style type="text/css" class="pd_save">
    .df-table-wrapper .panel-heading {
      border-radius: 0;
      padding: 0px;
    }
    .df-table-wrapper .panel-heading:hover {
      border-color: #008571;
    }
    .df-table-wrapper .panel-title a {
      background-color: #f9f9fb;
      color: #333333;
      display: block;
      outline: none;
      padding: 10px 15px;
      text-decoration: none;
    }
    .df-table-wrapper .panel-title a:hover {
      background-color: #337ab7;
      border-color: #2e6da4;
      color: #ffffff;
      display: block;
      padding: 10px 15px;
      text-decoration: none;
    }
    .df-table-wrapper {
      font-size: small;
      font-weight: 300;
      letter-spacing: 0.5px;
      line-height: normal;
      height: inherit;
      overflow: auto;
    }
    .df-table-search {
      margin: 0 0 20px 0;
    }
    .df-table-search-count {
      display: inline-block;
      margin: 0 0 20px 0;
    }
    .df-table-container {
      max-height: 50vh;
      max-width: 100%;
      overflow-x: auto;
      position: relative;
    }
    .df-table-wrapper table {
      border: 0 none #ffffff;
      border-collapse: collapse;
      margin: 0;
      min-width: 100%;
      padding: 0;
      table-layout: fixed;
      height: inherit;
      overflow: auto;
    }
    .df-table-wrapper tr.hidden {
      display: none;
    }
    .df-table-wrapper tr:nth-child(even) {
      background-color: #f9f9fb;
    }
    .df-table-wrapper tr.even {
      background-color: #f9f9fb;
    }
    .df-table-wrapper tr.odd {
      background-color: #ffffff;
    }
    .df-table-wrapper td + td {
      border-left: 1px solid #e0e0e0;
    }
  
    .df-table-wrapper thead,
    .fixed-header {
      font-weight: 600;
    }
    .df-table-wrapper tr,
    .fixed-row {
      border: 0 none #ffffff;
      margin: 0;
      padding: 0;
    }
    .df-table-wrapper th,
    .df-table-wrapper td,
    .fixed-cell {
      border: 0 none #ffffff;
      margin: 0;
      min-width: 50px;
      padding: 5px 20px 5px 10px;
      text-align: left;
      word-wrap: break-word;
    }
    .df-table-wrapper th {
      padding-bottom: 0;
      padding-top: 0;
    }
    .df-table-wrapper th div {
      max-height: 1px;
      visibility: hidden;
    }
  
    .df-schema-field {
      margin-left: 10px;
    }
  
    .fixed-header-container {
      overflow: hidden;
      position: relative;
    }
    .fixed-header {
      border-bottom: 2px solid #000;
      display: table;
      position: relative;
    }
    .fixed-row {
      display: table-row;
    }
    .fixed-cell {
      display: table-cell;
    }
  </style>
  
  
  <div class="df-table-wrapper df-table-wrapper-c58ef392 panel-group pd_save">
    <!-- dataframe schema -->
    
    <div class="panel panel-default">
      <div class="panel-heading">
        <h4 class="panel-title" style="margin: 0px;">
          <a data-toggle="collapse" href="#df-schema-c58ef392" data-parent="#df-table-wrapper-c58ef392">Schema</a>
        </h4>
      </div>
      <div id="df-schema-c58ef392" class="panel-collapse collapse">
        <div class="panel-body" style="font-family: monospace;">
          <div class="df-schema-fields">
            <div>Field types:</div>
            
              <div class="df-schema-field"><strong>artist: </strong> object</div>
            
              <div class="df-schema-field"><strong>auth: </strong> object</div>
            
              <div class="df-schema-field"><strong>firstName: </strong> object</div>
            
              <div class="df-schema-field"><strong>gender: </strong> object</div>
            
              <div class="df-schema-field"><strong>itemInSession: </strong> int64</div>
            
              <div class="df-schema-field"><strong>lastName: </strong> object</div>
            
              <div class="df-schema-field"><strong>length: </strong> float64</div>
            
              <div class="df-schema-field"><strong>level: </strong> object</div>
            
              <div class="df-schema-field"><strong>location: </strong> object</div>
            
              <div class="df-schema-field"><strong>method: </strong> object</div>
            
              <div class="df-schema-field"><strong>page: </strong> object</div>
            
              <div class="df-schema-field"><strong>registration: </strong> int64</div>
            
              <div class="df-schema-field"><strong>sessionId: </strong> int64</div>
            
              <div class="df-schema-field"><strong>song: </strong> object</div>
            
              <div class="df-schema-field"><strong>status: </strong> int64</div>
            
              <div class="df-schema-field"><strong>ts: </strong> int64</div>
            
              <div class="df-schema-field"><strong>userAgent: </strong> object</div>
            
              <div class="df-schema-field"><strong>userId: </strong> object</div>
            
              <div class="df-schema-field"><strong>churn: </strong> int32</div>
            
              <div class="df-schema-field"><strong>CSA: </strong> object</div>
            
              <div class="df-schema-field"><strong>ts_ts: </strong> datetime64[ns]</div>
            
              <div class="df-schema-field"><strong>ts_hour: </strong> int32</div>
            
              <div class="df-schema-field"><strong>ts_date: </strong> datetime64[ns]</div>
            
              <div class="df-schema-field"><strong>downgraded: </strong> int32</div>
            
              <div class="df-schema-field"><strong>upgraded: </strong> int32</div>
            
              <div class="df-schema-field"><strong>advert_shown: </strong> int32</div>
            
              <div class="df-schema-field"><strong>thumps_down: </strong> int32</div>
            
              <div class="df-schema-field"><strong>thumps_up: </strong> int32</div>
            
              <div class="df-schema-field"><strong>friend_added: </strong> int32</div>
            
              <div class="df-schema-field"><strong>song_added: </strong> int32</div>
            
              <div class="df-schema-field"><strong>gender_bin: </strong> int32</div>
            
              <div class="df-schema-field"><strong>level_bin: </strong> int32</div>
            
          </div>
        </div>
      </div>
    </div>
    
    <!-- dataframe table -->
    <div class="panel panel-default">
      
      <div class="panel-heading">
        <h4 class="panel-title" style="margin: 0px;">
          <a data-toggle="collapse" href="#df-table-c58ef392" data-parent="#df-table-wrapper-c58ef392"> Table</a>
        </h4>
      </div>
      
      <div id="df-table-c58ef392" class="panel-collapse collapse in">
        <div class="panel-body">
          
          <input type="text" class="df-table-search form-control input-sm" placeholder="Search table">
          
          <div>
            
            <span class="df-table-search-count">Showing 100 of 528005 rows</span>
            
          </div>
          <!-- fixed header for when dataframe table scrolls -->
          <div class="fixed-header-container">
            <div class="fixed-header">
              <div class="fixed-row">
                
                <div class="fixed-cell">artist</div>
                
                <div class="fixed-cell">auth</div>
                
                <div class="fixed-cell">firstName</div>
                
                <div class="fixed-cell">gender</div>
                
                <div class="fixed-cell">itemInSession</div>
                
                <div class="fixed-cell">lastName</div>
                
                <div class="fixed-cell">length</div>
                
                <div class="fixed-cell">level</div>
                
                <div class="fixed-cell">location</div>
                
                <div class="fixed-cell">method</div>
                
                <div class="fixed-cell">page</div>
                
                <div class="fixed-cell">registration</div>
                
                <div class="fixed-cell">sessionId</div>
                
                <div class="fixed-cell">song</div>
                
                <div class="fixed-cell">status</div>
                
                <div class="fixed-cell">ts</div>
                
                <div class="fixed-cell">userAgent</div>
                
                <div class="fixed-cell">userId</div>
                
                <div class="fixed-cell">churn</div>
                
                <div class="fixed-cell">CSA</div>
                
                <div class="fixed-cell">ts_ts</div>
                
                <div class="fixed-cell">ts_hour</div>
                
                <div class="fixed-cell">ts_date</div>
                
                <div class="fixed-cell">downgraded</div>
                
                <div class="fixed-cell">upgraded</div>
                
                <div class="fixed-cell">advert_shown</div>
                
                <div class="fixed-cell">thumps_down</div>
                
                <div class="fixed-cell">thumps_up</div>
                
                <div class="fixed-cell">friend_added</div>
                
                <div class="fixed-cell">song_added</div>
                
                <div class="fixed-cell">gender_bin</div>
                
                <div class="fixed-cell">level_bin</div>
                
              </div>
            </div>
          </div>
          <div class="df-table-container">
            <table class="df-table">
              <thead>
                <tr>
                  
                  <th><div>artist</div></th>
                  
                  <th><div>auth</div></th>
                  
                  <th><div>firstName</div></th>
                  
                  <th><div>gender</div></th>
                  
                  <th><div>itemInSession</div></th>
                  
                  <th><div>lastName</div></th>
                  
                  <th><div>length</div></th>
                  
                  <th><div>level</div></th>
                  
                  <th><div>location</div></th>
                  
                  <th><div>method</div></th>
                  
                  <th><div>page</div></th>
                  
                  <th><div>registration</div></th>
                  
                  <th><div>sessionId</div></th>
                  
                  <th><div>song</div></th>
                  
                  <th><div>status</div></th>
                  
                  <th><div>ts</div></th>
                  
                  <th><div>userAgent</div></th>
                  
                  <th><div>userId</div></th>
                  
                  <th><div>churn</div></th>
                  
                  <th><div>CSA</div></th>
                  
                  <th><div>ts_ts</div></th>
                  
                  <th><div>ts_hour</div></th>
                  
                  <th><div>ts_date</div></th>
                  
                  <th><div>downgraded</div></th>
                  
                  <th><div>upgraded</div></th>
                  
                  <th><div>advert_shown</div></th>
                  
                  <th><div>thumps_down</div></th>
                  
                  <th><div>thumps_up</div></th>
                  
                  <th><div>friend_added</div></th>
                  
                  <th><div>song_added</div></th>
                  
                  <th><div>gender_bin</div></th>
                  
                  <th><div>level_bin</div></th>
                  
                </tr>
              </thead>
              <tbody>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Sawyer</td>
                  
                  <td>M</td>
                  
                  <td>79</td>
                  
                  <td>Larson</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Houston-The Woodlands-Sugar Land, TX</td>
                  
                  <td>GET</td>
                  
                  <td>Help</td>
                  
                  <td>1538069638000</td>
                  
                  <td>97</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538352947000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>98</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 00:15:47</td>
                  
                  <td>0</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Bob James and David Sanborn</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>31</td>
                  
                  <td>Campos</td>
                  
                  <td>274.6771</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000</td>
                  
                  <td>245</td>
                  
                  <td>Maputo (Edit Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538353579000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                  <td>0</td>
                  
                  <td>AL</td>
                  
                  <td>2018-10-01 00:26:19</td>
                  
                  <td>0</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Beirut</td>
                  
                  <td>Logged In</td>
                  
                  <td>Payton</td>
                  
                  <td>F</td>
                  
                  <td>94</td>
                  
                  <td>Campbell</td>
                  
                  <td>213.68118</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529027541000</td>
                  
                  <td>38</td>
                  
                  <td>La Llorona</td>
                  
                  <td>200</td>
                  
                  <td>1538353937000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"</td>
                  
                  <td>39</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 00:32:17</td>
                  
                  <td>0</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Joel</td>
                  
                  <td>M</td>
                  
                  <td>17</td>
                  
                  <td>Thomas</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>Add Friend</td>
                  
                  <td>1534248752000</td>
                  
                  <td>485</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538358281000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.78.2 (KHTML, like Gecko) Version/7.0.6 Safari/537.78.2"</td>
                  
                  <td>273</td>
                  
                  <td>0</td>
                  
                  <td>NY-NJ-PA</td>
                  
                  <td>2018-10-01 01:44:41</td>
                  
                  <td>1</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Metallica</td>
                  
                  <td>Logged In</td>
                  
                  <td>Anthony</td>
                  
                  <td>M</td>
                  
                  <td>2</td>
                  
                  <td>Reed</td>
                  
                  <td>387.02975</td>
                  
                  <td>free</td>
                  
                  <td>Miami-Fort Lauderdale-West Palm Beach, FL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534823030000</td>
                  
                  <td>511</td>
                  
                  <td>Welcome Home (Sanitarium)</td>
                  
                  <td>200</td>
                  
                  <td>1538359083000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>166</td>
                  
                  <td>0</td>
                  
                  <td>FL</td>
                  
                  <td>2018-10-01 01:58:03</td>
                  
                  <td>1</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Aventura</td>
                  
                  <td>Logged In</td>
                  
                  <td>Carter</td>
                  
                  <td>M</td>
                  
                  <td>2</td>
                  
                  <td>Cook</td>
                  
                  <td>516.38812</td>
                  
                  <td>paid</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1522793334000</td>
                  
                  <td>287</td>
                  
                  <td>La Guerra</td>
                  
                  <td>200</td>
                  
                  <td>1538359362000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>288</td>
                  
                  <td>0</td>
                  
                  <td>IL-IN-WI</td>
                  
                  <td>2018-10-01 02:02:42</td>
                  
                  <td>2</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>DJ Khaled</td>
                  
                  <td>Logged In</td>
                  
                  <td>Maverick</td>
                  
                  <td>M</td>
                  
                  <td>222</td>
                  
                  <td>Santiago</td>
                  
                  <td>227.082</td>
                  
                  <td>paid</td>
                  
                  <td>Orlando-Kissimmee-Sanford, FL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535953455000</td>
                  
                  <td>178</td>
                  
                  <td>All I Do Is Win (feat. T-Pain_ Ludacris_ Snoop Dogg &amp; Rick Ross)</td>
                  
                  <td>200</td>
                  
                  <td>1538359711000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>179</td>
                  
                  <td>0</td>
                  
                  <td>FL</td>
                  
                  <td>2018-10-01 02:08:31</td>
                  
                  <td>2</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Luther Vandross</td>
                  
                  <td>Logged In</td>
                  
                  <td>Anthony</td>
                  
                  <td>M</td>
                  
                  <td>19</td>
                  
                  <td>Reed</td>
                  
                  <td>372.74077</td>
                  
                  <td>free</td>
                  
                  <td>Miami-Fort Lauderdale-West Palm Beach, FL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534823030000</td>
                  
                  <td>511</td>
                  
                  <td>The Glow Of Love</td>
                  
                  <td>200</td>
                  
                  <td>1538361807000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>166</td>
                  
                  <td>0</td>
                  
                  <td>FL</td>
                  
                  <td>2018-10-01 02:43:27</td>
                  
                  <td>2</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>82</td>
                  
                  <td>Campos</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1535931018000</td>
                  
                  <td>245</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538363212000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                  <td>0</td>
                  
                  <td>AL</td>
                  
                  <td>2018-10-01 03:06:52</td>
                  
                  <td>3</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Roudoudou</td>
                  
                  <td>Logged In</td>
                  
                  <td>Sofia</td>
                  
                  <td>F</td>
                  
                  <td>318</td>
                  
                  <td>Gordon</td>
                  
                  <td>267.93751</td>
                  
                  <td>paid</td>
                  
                  <td>Rochester, MN</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533175710000</td>
                  
                  <td>162</td>
                  
                  <td>Zoom Zoom</td>
                  
                  <td>200</td>
                  
                  <td>1538363565000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>163</td>
                  
                  <td>0</td>
                  
                  <td>MN</td>
                  
                  <td>2018-10-01 03:12:45</td>
                  
                  <td>3</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>9</td>
                  
                  <td>Roberts</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1537634865000</td>
                  
                  <td>27</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538363570000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                  <td>0</td>
                  
                  <td>OH</td>
                  
                  <td>2018-10-01 03:12:50</td>
                  
                  <td>3</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Natiruts</td>
                  
                  <td>Logged In</td>
                  
                  <td>Aurora</td>
                  
                  <td>F</td>
                  
                  <td>3</td>
                  
                  <td>Humphrey</td>
                  
                  <td>200.64608</td>
                  
                  <td>paid</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536795126000</td>
                  
                  <td>418</td>
                  
                  <td>Bob Falou</td>
                  
                  <td>200</td>
                  
                  <td>1538363728000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>127</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 03:15:28</td>
                  
                  <td>3</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Barry Tuckwell/Academy of St Martin-in-the-Fields/Sir Neville Marriner</td>
                  
                  <td>Logged In</td>
                  
                  <td>Anthony</td>
                  
                  <td>M</td>
                  
                  <td>26</td>
                  
                  <td>Reed</td>
                  
                  <td>277.15873</td>
                  
                  <td>free</td>
                  
                  <td>Miami-Fort Lauderdale-West Palm Beach, FL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534823030000</td>
                  
                  <td>511</td>
                  
                  <td>Horn Concerto No. 4 in E flat K495: II. Romance (Andante cantabile)</td>
                  
                  <td>200</td>
                  
                  <td>1538364246000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>166</td>
                  
                  <td>0</td>
                  
                  <td>FL</td>
                  
                  <td>2018-10-01 03:24:06</td>
                  
                  <td>3</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Mariah Carey</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>26</td>
                  
                  <td>Roberts</td>
                  
                  <td>216.29342</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537634865000</td>
                  
                  <td>27</td>
                  
                  <td>Last Kiss</td>
                  
                  <td>200</td>
                  
                  <td>1538366846000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                  <td>0</td>
                  
                  <td>OH</td>
                  
                  <td>2018-10-01 04:07:26</td>
                  
                  <td>4</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Linkin Park</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>102</td>
                  
                  <td>Campos</td>
                  
                  <td>494.99383</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000</td>
                  
                  <td>245</td>
                  
                  <td>Bleed It Out [Live At Milton Keynes]</td>
                  
                  <td>200</td>
                  
                  <td>1538367186000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                  <td>0</td>
                  
                  <td>AL</td>
                  
                  <td>2018-10-01 04:13:06</td>
                  
                  <td>4</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Martin Jondo</td>
                  
                  <td>Logged In</td>
                  
                  <td>Carter</td>
                  
                  <td>M</td>
                  
                  <td>45</td>
                  
                  <td>Cook</td>
                  
                  <td>221.41342</td>
                  
                  <td>paid</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1522793334000</td>
                  
                  <td>287</td>
                  
                  <td>Clearly feat. Gentleman</td>
                  
                  <td>200</td>
                  
                  <td>1538368101000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>288</td>
                  
                  <td>0</td>
                  
                  <td>IL-IN-WI</td>
                  
                  <td>2018-10-01 04:28:21</td>
                  
                  <td>4</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Sunset Rubdown</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jhaden</td>
                  
                  <td>M</td>
                  
                  <td>33</td>
                  
                  <td>Cooper</td>
                  
                  <td>143.59465</td>
                  
                  <td>free</td>
                  
                  <td>Knoxville, TN</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536859413000</td>
                  
                  <td>249</td>
                  
                  <td>Setting vs. Rising</td>
                  
                  <td>200</td>
                  
                  <td>1538368231000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>250</td>
                  
                  <td>0</td>
                  
                  <td>TN</td>
                  
                  <td>2018-10-01 04:30:31</td>
                  
                  <td>4</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Iron Maiden</td>
                  
                  <td>Logged In</td>
                  
                  <td>Adriel</td>
                  
                  <td>M</td>
                  
                  <td>5</td>
                  
                  <td>Mendoza</td>
                  
                  <td>409.5473</td>
                  
                  <td>paid</td>
                  
                  <td>Kansas City, MO-KS</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535623466000</td>
                  
                  <td>476</td>
                  
                  <td>Ghost Of The Navigator</td>
                  
                  <td>200</td>
                  
                  <td>1538370333000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"</td>
                  
                  <td>18</td>
                  
                  <td>0</td>
                  
                  <td>MO-KS</td>
                  
                  <td>2018-10-01 05:05:33</td>
                  
                  <td>5</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Tea Leaf Green</td>
                  
                  <td>Logged In</td>
                  
                  <td>Maverick</td>
                  
                  <td>M</td>
                  
                  <td>279</td>
                  
                  <td>Santiago</td>
                  
                  <td>327.91465</td>
                  
                  <td>paid</td>
                  
                  <td>Orlando-Kissimmee-Sanford, FL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535953455000</td>
                  
                  <td>178</td>
                  
                  <td>Taught To Be Proud (Rock)</td>
                  
                  <td>200</td>
                  
                  <td>1538371686000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>179</td>
                  
                  <td>0</td>
                  
                  <td>FL</td>
                  
                  <td>2018-10-01 05:28:06</td>
                  
                  <td>5</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Cut Copy</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>59</td>
                  
                  <td>Roberts</td>
                  
                  <td>292.5971</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537634865000</td>
                  
                  <td>27</td>
                  
                  <td>Hearts On Fire</td>
                  
                  <td>200</td>
                  
                  <td>1538372684000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                  <td>0</td>
                  
                  <td>OH</td>
                  
                  <td>2018-10-01 05:44:44</td>
                  
                  <td>5</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Fluke</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jhaden</td>
                  
                  <td>M</td>
                  
                  <td>58</td>
                  
                  <td>Cooper</td>
                  
                  <td>348.89098</td>
                  
                  <td>free</td>
                  
                  <td>Knoxville, TN</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536859413000</td>
                  
                  <td>249</td>
                  
                  <td>Absurd</td>
                  
                  <td>200</td>
                  
                  <td>1538373578000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>250</td>
                  
                  <td>0</td>
                  
                  <td>TN</td>
                  
                  <td>2018-10-01 05:59:38</td>
                  
                  <td>5</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Gino D'Auri</td>
                  
                  <td>Logged In</td>
                  
                  <td>Carter</td>
                  
                  <td>M</td>
                  
                  <td>74</td>
                  
                  <td>Cook</td>
                  
                  <td>334.70649</td>
                  
                  <td>paid</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1522793334000</td>
                  
                  <td>287</td>
                  
                  <td>Quelo de Triana (Soleares)</td>
                  
                  <td>200</td>
                  
                  <td>1538374718000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>288</td>
                  
                  <td>0</td>
                  
                  <td>IL-IN-WI</td>
                  
                  <td>2018-10-01 06:18:38</td>
                  
                  <td>6</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Oliver</td>
                  
                  <td>M</td>
                  
                  <td>38</td>
                  
                  <td>Fry</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>Add to Playlist</td>
                  
                  <td>1538048434000</td>
                  
                  <td>153</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538375152000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>154</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 06:25:52</td>
                  
                  <td>6</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Aurora</td>
                  
                  <td>F</td>
                  
                  <td>62</td>
                  
                  <td>Humphrey</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1536795126000</td>
                  
                  <td>418</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538375183000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>127</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 06:26:23</td>
                  
                  <td>6</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Dashboard Confessional</td>
                  
                  <td>Logged In</td>
                  
                  <td>Oliver</td>
                  
                  <td>M</td>
                  
                  <td>40</td>
                  
                  <td>Fry</td>
                  
                  <td>226.29832</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538048434000</td>
                  
                  <td>153</td>
                  
                  <td>Screaming Infidelities</td>
                  
                  <td>200</td>
                  
                  <td>1538375479000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>154</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 06:31:19</td>
                  
                  <td>6</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Red Jumpsuit Apparatus</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jessiah</td>
                  
                  <td>M</td>
                  
                  <td>16</td>
                  
                  <td>Rose</td>
                  
                  <td>191.84281</td>
                  
                  <td>free</td>
                  
                  <td>Richmond, VA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532413080000</td>
                  
                  <td>529</td>
                  
                  <td>Face Down (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538375653000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>207</td>
                  
                  <td>0</td>
                  
                  <td>VA</td>
                  
                  <td>2018-10-01 06:34:13</td>
                  
                  <td>6</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Ismo Alanko SÃÂ¤ÃÂ¤tiÃÂ¶</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>78</td>
                  
                  <td>Roberts</td>
                  
                  <td>218.93179</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537634865000</td>
                  
                  <td>27</td>
                  
                  <td>Nokian Takana</td>
                  
                  <td>200</td>
                  
                  <td>1538376939000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                  <td>0</td>
                  
                  <td>OH</td>
                  
                  <td>2018-10-01 06:55:39</td>
                  
                  <td>6</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Keely Smith</td>
                  
                  <td>Logged In</td>
                  
                  <td>Adriel</td>
                  
                  <td>M</td>
                  
                  <td>41</td>
                  
                  <td>Mendoza</td>
                  
                  <td>156.60363</td>
                  
                  <td>paid</td>
                  
                  <td>Kansas City, MO-KS</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535623466000</td>
                  
                  <td>476</td>
                  
                  <td>You Go To My Head</td>
                  
                  <td>200</td>
                  
                  <td>1538377618000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"</td>
                  
                  <td>18</td>
                  
                  <td>0</td>
                  
                  <td>MO-KS</td>
                  
                  <td>2018-10-01 07:06:58</td>
                  
                  <td>7</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Carter</td>
                  
                  <td>M</td>
                  
                  <td>91</td>
                  
                  <td>Cook</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>GET</td>
                  
                  <td>Downgrade</td>
                  
                  <td>1522793334000</td>
                  
                  <td>287</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538378190000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>288</td>
                  
                  <td>0</td>
                  
                  <td>IL-IN-WI</td>
                  
                  <td>2018-10-01 07:16:30</td>
                  
                  <td>7</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Phoenix</td>
                  
                  <td>Logged In</td>
                  
                  <td>Adriel</td>
                  
                  <td>M</td>
                  
                  <td>50</td>
                  
                  <td>Mendoza</td>
                  
                  <td>184.73751</td>
                  
                  <td>paid</td>
                  
                  <td>Kansas City, MO-KS</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535623466000</td>
                  
                  <td>476</td>
                  
                  <td>Long Distance Call</td>
                  
                  <td>200</td>
                  
                  <td>1538379064000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"</td>
                  
                  <td>18</td>
                  
                  <td>0</td>
                  
                  <td>MO-KS</td>
                  
                  <td>2018-10-01 07:31:04</td>
                  
                  <td>7</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Maverick</td>
                  
                  <td>M</td>
                  
                  <td>319</td>
                  
                  <td>Santiago</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Orlando-Kissimmee-Sanford, FL</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1535953455000</td>
                  
                  <td>178</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538379124000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>179</td>
                  
                  <td>0</td>
                  
                  <td>FL</td>
                  
                  <td>2018-10-01 07:32:04</td>
                  
                  <td>7</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Flyleaf</td>
                  
                  <td>Logged In</td>
                  
                  <td>Adriel</td>
                  
                  <td>M</td>
                  
                  <td>52</td>
                  
                  <td>Mendoza</td>
                  
                  <td>177.99791</td>
                  
                  <td>paid</td>
                  
                  <td>Kansas City, MO-KS</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535623466000</td>
                  
                  <td>476</td>
                  
                  <td>Cassie</td>
                  
                  <td>200</td>
                  
                  <td>1538379544000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"</td>
                  
                  <td>18</td>
                  
                  <td>0</td>
                  
                  <td>MO-KS</td>
                  
                  <td>2018-10-01 07:39:04</td>
                  
                  <td>7</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Jason Mraz &amp; Colbie Caillat</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nora</td>
                  
                  <td>F</td>
                  
                  <td>31</td>
                  
                  <td>Kennedy</td>
                  
                  <td>189.6224</td>
                  
                  <td>free</td>
                  
                  <td>Madison, WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533944611000</td>
                  
                  <td>524</td>
                  
                  <td>Lucky (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538380423000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>301</td>
                  
                  <td>0</td>
                  
                  <td>WI</td>
                  
                  <td>2018-10-01 07:53:43</td>
                  
                  <td>7</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Chris Brown</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ainsley</td>
                  
                  <td>F</td>
                  
                  <td>27</td>
                  
                  <td>Farley</td>
                  
                  <td>202.23955</td>
                  
                  <td>free</td>
                  
                  <td>McAllen-Edinburg-Mission, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538304455000</td>
                  
                  <td>499</td>
                  
                  <td>You</td>
                  
                  <td>200</td>
                  
                  <td>1538380700000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>78</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 07:58:20</td>
                  
                  <td>7</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Tub Ring</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>185</td>
                  
                  <td>Campos</td>
                  
                  <td>233.69098</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000</td>
                  
                  <td>245</td>
                  
                  <td>Invalid</td>
                  
                  <td>200</td>
                  
                  <td>1538381417000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                  <td>0</td>
                  
                  <td>AL</td>
                  
                  <td>2018-10-01 08:10:17</td>
                  
                  <td>8</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Eyedea &amp; Abilities</td>
                  
                  <td>Logged In</td>
                  
                  <td>Faigy</td>
                  
                  <td>F</td>
                  
                  <td>45</td>
                  
                  <td>Howe</td>
                  
                  <td>271.25506</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538211832000</td>
                  
                  <td>492</td>
                  
                  <td>Glass</td>
                  
                  <td>200</td>
                  
                  <td>1538382133000</td>
                  
                  <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>95</td>
                  
                  <td>0</td>
                  
                  <td>AZ</td>
                  
                  <td>2018-10-01 08:22:13</td>
                  
                  <td>8</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Pamela Williams</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jhaden</td>
                  
                  <td>M</td>
                  
                  <td>112</td>
                  
                  <td>Cooper</td>
                  
                  <td>289.27955</td>
                  
                  <td>free</td>
                  
                  <td>Knoxville, TN</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536859413000</td>
                  
                  <td>249</td>
                  
                  <td>Escape To Paradise</td>
                  
                  <td>200</td>
                  
                  <td>1538385589000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>250</td>
                  
                  <td>0</td>
                  
                  <td>TN</td>
                  
                  <td>2018-10-01 09:19:49</td>
                  
                  <td>9</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Slim Dusty</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jhaden</td>
                  
                  <td>M</td>
                  
                  <td>114</td>
                  
                  <td>Cooper</td>
                  
                  <td>198.922</td>
                  
                  <td>free</td>
                  
                  <td>Knoxville, TN</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536859413000</td>
                  
                  <td>249</td>
                  
                  <td>Long Black Road</td>
                  
                  <td>200</td>
                  
                  <td>1538386158000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>250</td>
                  
                  <td>0</td>
                  
                  <td>TN</td>
                  
                  <td>2018-10-01 09:29:18</td>
                  
                  <td>9</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Paramore</td>
                  
                  <td>Logged In</td>
                  
                  <td>Evan</td>
                  
                  <td>M</td>
                  
                  <td>9</td>
                  
                  <td>Shelton</td>
                  
                  <td>226.29832</td>
                  
                  <td>free</td>
                  
                  <td>Hagerstown-Martinsburg, MD-WV</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534894284000</td>
                  
                  <td>479</td>
                  
                  <td>Here We Go Again (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538387734000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>251</td>
                  
                  <td>0</td>
                  
                  <td>MD-WV</td>
                  
                  <td>2018-10-01 09:55:34</td>
                  
                  <td>9</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Teitur</td>
                  
                  <td>Logged In</td>
                  
                  <td>Daelin</td>
                  
                  <td>M</td>
                  
                  <td>7</td>
                  
                  <td>Turner</td>
                  
                  <td>344.52853</td>
                  
                  <td>free</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538227408000</td>
                  
                  <td>125</td>
                  
                  <td>Guilt By Association</td>
                  
                  <td>200</td>
                  
                  <td>1538388640000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>126</td>
                  
                  <td>0</td>
                  
                  <td>NY-NJ-PA</td>
                  
                  <td>2018-10-01 10:10:40</td>
                  
                  <td>10</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Brayden</td>
                  
                  <td>M</td>
                  
                  <td>2</td>
                  
                  <td>Thomas</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1534133898000</td>
                  
                  <td>498</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538388789000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>85</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 10:13:09</td>
                  
                  <td>10</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Infected Mushroom</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jackson</td>
                  
                  <td>M</td>
                  
                  <td>6</td>
                  
                  <td>Hoffman</td>
                  
                  <td>323.63057</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537054964000</td>
                  
                  <td>184</td>
                  
                  <td>Drop Out</td>
                  
                  <td>200</td>
                  
                  <td>1538390274000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>185</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 10:37:54</td>
                  
                  <td>10</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Evan</td>
                  
                  <td>M</td>
                  
                  <td>32</td>
                  
                  <td>Shelton</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Hagerstown-Martinsburg, MD-WV</td>
                  
                  <td>PUT</td>
                  
                  <td>Add to Playlist</td>
                  
                  <td>1534894284000</td>
                  
                  <td>479</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538391880000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>251</td>
                  
                  <td>0</td>
                  
                  <td>MD-WV</td>
                  
                  <td>2018-10-01 11:04:40</td>
                  
                  <td>11</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>J.J. Cale</td>
                  
                  <td>Logged In</td>
                  
                  <td>Daelin</td>
                  
                  <td>M</td>
                  
                  <td>33</td>
                  
                  <td>Turner</td>
                  
                  <td>161.12281</td>
                  
                  <td>free</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538227408000</td>
                  
                  <td>125</td>
                  
                  <td>Runaround</td>
                  
                  <td>200</td>
                  
                  <td>1538392902000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>126</td>
                  
                  <td>0</td>
                  
                  <td>NY-NJ-PA</td>
                  
                  <td>2018-10-01 11:21:42</td>
                  
                  <td>11</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Streets</td>
                  
                  <td>Logged In</td>
                  
                  <td>Daelin</td>
                  
                  <td>M</td>
                  
                  <td>39</td>
                  
                  <td>Turner</td>
                  
                  <td>208.92689</td>
                  
                  <td>free</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538227408000</td>
                  
                  <td>125</td>
                  
                  <td>The Irony Of It All (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538394116000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>126</td>
                  
                  <td>0</td>
                  
                  <td>NY-NJ-PA</td>
                  
                  <td>2018-10-01 11:41:56</td>
                  
                  <td>11</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Cut Copy</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jariel</td>
                  
                  <td>M</td>
                  
                  <td>4</td>
                  
                  <td>Barber</td>
                  
                  <td>328.12363</td>
                  
                  <td>free</td>
                  
                  <td>Vermillion, SD</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535301187000</td>
                  
                  <td>318</td>
                  
                  <td>Going Nowhere</td>
                  
                  <td>200</td>
                  
                  <td>1538395196000</td>
                  
                  <td>"Mozilla/5.0 (iPad; CPU OS 7_1_1 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D201 Safari/9537.53"</td>
                  
                  <td>192</td>
                  
                  <td>0</td>
                  
                  <td>SD</td>
                  
                  <td>2018-10-01 11:59:56</td>
                  
                  <td>11</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Angelspit</td>
                  
                  <td>Logged In</td>
                  
                  <td>Faigy</td>
                  
                  <td>F</td>
                  
                  <td>117</td>
                  
                  <td>Howe</td>
                  
                  <td>280.78975</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538211832000</td>
                  
                  <td>492</td>
                  
                  <td>Get Even</td>
                  
                  <td>200</td>
                  
                  <td>1538397445000</td>
                  
                  <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>95</td>
                  
                  <td>0</td>
                  
                  <td>AZ</td>
                  
                  <td>2018-10-01 12:37:25</td>
                  
                  <td>12</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>N.E.R.D.</td>
                  
                  <td>Logged In</td>
                  
                  <td>Joseph</td>
                  
                  <td>M</td>
                  
                  <td>4</td>
                  
                  <td>Morales</td>
                  
                  <td>259.23873</td>
                  
                  <td>free</td>
                  
                  <td>Corpus Christi, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532063507000</td>
                  
                  <td>526</td>
                  
                  <td>Rock Star</td>
                  
                  <td>200</td>
                  
                  <td>1538400054000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>293</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 13:20:54</td>
                  
                  <td>13</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Joseph</td>
                  
                  <td>M</td>
                  
                  <td>7</td>
                  
                  <td>Morales</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Corpus Christi, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>Logout</td>
                  
                  <td>1532063507000</td>
                  
                  <td>526</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538400541000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>293</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 13:29:01</td>
                  
                  <td>13</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Muse</td>
                  
                  <td>Logged In</td>
                  
                  <td>Faigy</td>
                  
                  <td>F</td>
                  
                  <td>138</td>
                  
                  <td>Howe</td>
                  
                  <td>255.08526</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538211832000</td>
                  
                  <td>492</td>
                  
                  <td>Do We Need This?</td>
                  
                  <td>200</td>
                  
                  <td>1538401091000</td>
                  
                  <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>95</td>
                  
                  <td>0</td>
                  
                  <td>AZ</td>
                  
                  <td>2018-10-01 13:38:11</td>
                  
                  <td>13</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Dierks Bentley</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>42</td>
                  
                  <td>Campos</td>
                  
                  <td>272.66567</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000</td>
                  
                  <td>548</td>
                  
                  <td>Lot Of Leavin' Left To Do</td>
                  
                  <td>200</td>
                  
                  <td>1538405297000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                  <td>0</td>
                  
                  <td>AL</td>
                  
                  <td>2018-10-01 14:48:17</td>
                  
                  <td>14</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>La Mancha De Rolando</td>
                  
                  <td>Logged In</td>
                  
                  <td>Anna</td>
                  
                  <td>F</td>
                  
                  <td>26</td>
                  
                  <td>Williams</td>
                  
                  <td>243.98322</td>
                  
                  <td>free</td>
                  
                  <td>Charlotte-Concord-Gastonia, NC-SC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1526838391000</td>
                  
                  <td>425</td>
                  
                  <td>Melodia Simple</td>
                  
                  <td>200</td>
                  
                  <td>1538405967000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>119</td>
                  
                  <td>0</td>
                  
                  <td>NC-SC</td>
                  
                  <td>2018-10-01 14:59:27</td>
                  
                  <td>14</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Roxette</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lucero</td>
                  
                  <td>F</td>
                  
                  <td>2</td>
                  
                  <td>Reed</td>
                  
                  <td>279.92771</td>
                  
                  <td>free</td>
                  
                  <td>Louisville/Jefferson County, KY-IN</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536642109000</td>
                  
                  <td>139</td>
                  
                  <td>Un Dia Sin Ti (Spending My Time)</td>
                  
                  <td>200</td>
                  
                  <td>1538406476000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>140</td>
                  
                  <td>0</td>
                  
                  <td>KY-IN</td>
                  
                  <td>2018-10-01 15:07:56</td>
                  
                  <td>15</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Emily</td>
                  
                  <td>F</td>
                  
                  <td>14</td>
                  
                  <td>Morrison</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1534367797000</td>
                  
                  <td>477</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538406511000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>232</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 15:08:31</td>
                  
                  <td>15</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Every Time I Die</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>32</td>
                  
                  <td>Raymond</td>
                  
                  <td>172.64281</td>
                  
                  <td>free</td>
                  
                  <td>Hartford-West Hartford-East Hartford, CT</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534245996000</td>
                  
                  <td>26</td>
                  
                  <td>The New Black</td>
                  
                  <td>200</td>
                  
                  <td>1538406621000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                  <td>0</td>
                  
                  <td>CT</td>
                  
                  <td>2018-10-01 15:10:21</td>
                  
                  <td>15</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Leonard Cohen</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nicole</td>
                  
                  <td>F</td>
                  
                  <td>2</td>
                  
                  <td>Beck</td>
                  
                  <td>267.67628</td>
                  
                  <td>paid</td>
                  
                  <td>Vineland-Bridgeton, NJ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532224335000</td>
                  
                  <td>123</td>
                  
                  <td>I'm Your Man</td>
                  
                  <td>200</td>
                  
                  <td>1538407380000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                  <td>0</td>
                  
                  <td>NJ</td>
                  
                  <td>2018-10-01 15:23:00</td>
                  
                  <td>15</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Ray Scott</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>38</td>
                  
                  <td>Raymond</td>
                  
                  <td>178.52036</td>
                  
                  <td>free</td>
                  
                  <td>Hartford-West Hartford-East Hartford, CT</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534245996000</td>
                  
                  <td>26</td>
                  
                  <td>I Didn't Come Here To Talk (Radio Edit)</td>
                  
                  <td>200</td>
                  
                  <td>1538407770000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                  <td>0</td>
                  
                  <td>CT</td>
                  
                  <td>2018-10-01 15:29:30</td>
                  
                  <td>15</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Lonely Island / T-Pain</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nicole</td>
                  
                  <td>F</td>
                  
                  <td>4</td>
                  
                  <td>Beck</td>
                  
                  <td>156.23791</td>
                  
                  <td>paid</td>
                  
                  <td>Vineland-Bridgeton, NJ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532224335000</td>
                  
                  <td>123</td>
                  
                  <td>I'm On A Boat</td>
                  
                  <td>200</td>
                  
                  <td>1538407904000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                  <td>0</td>
                  
                  <td>NJ</td>
                  
                  <td>2018-10-01 15:31:44</td>
                  
                  <td>15</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Mark Hollis</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>60</td>
                  
                  <td>Campos</td>
                  
                  <td>490.1873</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000</td>
                  
                  <td>548</td>
                  
                  <td>A Life (1895 - 1915)</td>
                  
                  <td>200</td>
                  
                  <td>1538408633000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                  <td>0</td>
                  
                  <td>AL</td>
                  
                  <td>2018-10-01 15:43:53</td>
                  
                  <td>15</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Carbon Leaf</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nicole</td>
                  
                  <td>F</td>
                  
                  <td>14</td>
                  
                  <td>Beck</td>
                  
                  <td>207.62077</td>
                  
                  <td>paid</td>
                  
                  <td>Vineland-Bridgeton, NJ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532224335000</td>
                  
                  <td>123</td>
                  
                  <td>Life Less Ordinary</td>
                  
                  <td>200</td>
                  
                  <td>1538410398000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                  <td>0</td>
                  
                  <td>NJ</td>
                  
                  <td>2018-10-01 16:13:18</td>
                  
                  <td>16</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Cartola</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>53</td>
                  
                  <td>Raymond</td>
                  
                  <td>208.92689</td>
                  
                  <td>free</td>
                  
                  <td>Hartford-West Hartford-East Hartford, CT</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534245996000</td>
                  
                  <td>26</td>
                  
                  <td>Sala De RecepÃÂ§ÃÂ£o</td>
                  
                  <td>200</td>
                  
                  <td>1538411478000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                  <td>0</td>
                  
                  <td>CT</td>
                  
                  <td>2018-10-01 16:31:18</td>
                  
                  <td>16</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Kings Of Leon</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>86</td>
                  
                  <td>Campos</td>
                  
                  <td>201.79546</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000</td>
                  
                  <td>548</td>
                  
                  <td>Revelry</td>
                  
                  <td>200</td>
                  
                  <td>1538412791000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                  <td>0</td>
                  
                  <td>AL</td>
                  
                  <td>2018-10-01 16:53:11</td>
                  
                  <td>16</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Depeche Mode</td>
                  
                  <td>Logged In</td>
                  
                  <td>Adam</td>
                  
                  <td>M</td>
                  
                  <td>15</td>
                  
                  <td>Johnson</td>
                  
                  <td>227.05587</td>
                  
                  <td>free</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536986118000</td>
                  
                  <td>173</td>
                  
                  <td>Master And Servant</td>
                  
                  <td>200</td>
                  
                  <td>1538414280000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>174</td>
                  
                  <td>0</td>
                  
                  <td>IL-IN-WI</td>
                  
                  <td>2018-10-01 17:18:00</td>
                  
                  <td>17</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>tobyMac</td>
                  
                  <td>Logged In</td>
                  
                  <td>Annabella</td>
                  
                  <td>F</td>
                  
                  <td>10</td>
                  
                  <td>Knapp</td>
                  
                  <td>272.09098</td>
                  
                  <td>free</td>
                  
                  <td>San Antonio-New Braunfels, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535525247000</td>
                  
                  <td>39</td>
                  
                  <td>City On Our Knees</td>
                  
                  <td>200</td>
                  
                  <td>1538415104000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>40</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 17:31:44</td>
                  
                  <td>17</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Shinedown</td>
                  
                  <td>Logged In</td>
                  
                  <td>Annabella</td>
                  
                  <td>F</td>
                  
                  <td>22</td>
                  
                  <td>Knapp</td>
                  
                  <td>429.60934</td>
                  
                  <td>free</td>
                  
                  <td>San Antonio-New Braunfels, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535525247000</td>
                  
                  <td>39</td>
                  
                  <td>Lady So Devine (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538417126000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>40</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 18:05:26</td>
                  
                  <td>18</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Katy Perry</td>
                  
                  <td>Logged In</td>
                  
                  <td>Spencer</td>
                  
                  <td>M</td>
                  
                  <td>6</td>
                  
                  <td>Gonzalez</td>
                  
                  <td>179.40853</td>
                  
                  <td>free</td>
                  
                  <td>Concord, NH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537347211000</td>
                  
                  <td>64</td>
                  
                  <td>I Kissed A Girl</td>
                  
                  <td>200</td>
                  
                  <td>1538417725000</td>
                  
                  <td>Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>65</td>
                  
                  <td>0</td>
                  
                  <td>NH</td>
                  
                  <td>2018-10-01 18:15:25</td>
                  
                  <td>18</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>148</td>
                  
                  <td>Myers</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Grand Rapids-Wyoming, MI</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1529995579000</td>
                  
                  <td>235</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538417790000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>236</td>
                  
                  <td>0</td>
                  
                  <td>MI</td>
                  
                  <td>2018-10-01 18:16:30</td>
                  
                  <td>18</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Emily</td>
                  
                  <td>F</td>
                  
                  <td>80</td>
                  
                  <td>Morrison</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>GET</td>
                  
                  <td>About</td>
                  
                  <td>1534367797000</td>
                  
                  <td>477</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538417929000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>232</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 18:18:49</td>
                  
                  <td>18</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Spencer</td>
                  
                  <td>M</td>
                  
                  <td>12</td>
                  
                  <td>Gonzalez</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Concord, NH</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1537347211000</td>
                  
                  <td>64</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538418720000</td>
                  
                  <td>Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>65</td>
                  
                  <td>0</td>
                  
                  <td>NH</td>
                  
                  <td>2018-10-01 18:32:00</td>
                  
                  <td>18</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Q-Tip</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>18</td>
                  
                  <td>Roberts</td>
                  
                  <td>181.78567</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537634865000</td>
                  
                  <td>565</td>
                  
                  <td>You</td>
                  
                  <td>200</td>
                  
                  <td>1538418886000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                  <td>0</td>
                  
                  <td>OH</td>
                  
                  <td>2018-10-01 18:34:46</td>
                  
                  <td>18</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Paramore</td>
                  
                  <td>Logged In</td>
                  
                  <td>Thomas</td>
                  
                  <td>M</td>
                  
                  <td>1</td>
                  
                  <td>White</td>
                  
                  <td>267.65016</td>
                  
                  <td>free</td>
                  
                  <td>Providence-Warwick, RI-MA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536270348000</td>
                  
                  <td>171</td>
                  
                  <td>The Only Exception (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538418966000</td>
                  
                  <td>"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/36.0.1985.125 Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>172</td>
                  
                  <td>0</td>
                  
                  <td>RI-MA</td>
                  
                  <td>2018-10-01 18:36:06</td>
                  
                  <td>18</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Yes</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jasmine</td>
                  
                  <td>F</td>
                  
                  <td>39</td>
                  
                  <td>Richardson</td>
                  
                  <td>228.93669</td>
                  
                  <td>free</td>
                  
                  <td>Philadelphia-Camden-Wilmington, PA-NJ-DE-MD</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1531477817000</td>
                  
                  <td>166</td>
                  
                  <td>CLAP</td>
                  
                  <td>200</td>
                  
                  <td>1538420020000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>167</td>
                  
                  <td>0</td>
                  
                  <td>PA-NJ-DE-MD</td>
                  
                  <td>2018-10-01 18:53:40</td>
                  
                  <td>18</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lucero</td>
                  
                  <td>F</td>
                  
                  <td>38</td>
                  
                  <td>Reed</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Louisville/Jefferson County, KY-IN</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1536642109000</td>
                  
                  <td>570</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538420069000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>140</td>
                  
                  <td>0</td>
                  
                  <td>KY-IN</td>
                  
                  <td>2018-10-01 18:54:29</td>
                  
                  <td>18</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>New Radicals</td>
                  
                  <td>Logged In</td>
                  
                  <td>Chase</td>
                  
                  <td>M</td>
                  
                  <td>23</td>
                  
                  <td>Ross</td>
                  
                  <td>300.82567</td>
                  
                  <td>free</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532450666000</td>
                  
                  <td>136</td>
                  
                  <td>You Get What You Give</td>
                  
                  <td>200</td>
                  
                  <td>1538420777000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.78.2 (KHTML, like Gecko) Version/7.0.6 Safari/537.78.2"</td>
                  
                  <td>137</td>
                  
                  <td>0</td>
                  
                  <td>NY-NJ-PA</td>
                  
                  <td>2018-10-01 19:06:17</td>
                  
                  <td>19</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Emily</td>
                  
                  <td>F</td>
                  
                  <td>100</td>
                  
                  <td>Morrison</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>Add to Playlist</td>
                  
                  <td>1534367797000</td>
                  
                  <td>477</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538420831000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>232</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 19:07:11</td>
                  
                  <td>19</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Guasones</td>
                  
                  <td>Logged In</td>
                  
                  <td>Chase</td>
                  
                  <td>M</td>
                  
                  <td>33</td>
                  
                  <td>Ross</td>
                  
                  <td>275.09506</td>
                  
                  <td>free</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532450666000</td>
                  
                  <td>136</td>
                  
                  <td>Chica De Ojos Tristes</td>
                  
                  <td>200</td>
                  
                  <td>1538423254000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.78.2 (KHTML, like Gecko) Version/7.0.6 Safari/537.78.2"</td>
                  
                  <td>137</td>
                  
                  <td>0</td>
                  
                  <td>NY-NJ-PA</td>
                  
                  <td>2018-10-01 19:47:34</td>
                  
                  <td>19</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Bachman-Turner Overdrive</td>
                  
                  <td>Logged In</td>
                  
                  <td>Joseph</td>
                  
                  <td>M</td>
                  
                  <td>6</td>
                  
                  <td>Morales</td>
                  
                  <td>238.49751</td>
                  
                  <td>free</td>
                  
                  <td>Corpus Christi, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532063507000</td>
                  
                  <td>563</td>
                  
                  <td>Roll On Down The Highway</td>
                  
                  <td>200</td>
                  
                  <td>1538423941000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>293</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 19:59:01</td>
                  
                  <td>19</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>White Lion</td>
                  
                  <td>Logged In</td>
                  
                  <td>Joseph</td>
                  
                  <td>M</td>
                  
                  <td>7</td>
                  
                  <td>Morales</td>
                  
                  <td>450.55955</td>
                  
                  <td>free</td>
                  
                  <td>Corpus Christi, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532063507000</td>
                  
                  <td>563</td>
                  
                  <td>Wait</td>
                  
                  <td>200</td>
                  
                  <td>1538424179000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>293</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 20:02:59</td>
                  
                  <td>20</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Grouch &amp; Eligh</td>
                  
                  <td>Logged In</td>
                  
                  <td>Joseph</td>
                  
                  <td>M</td>
                  
                  <td>14</td>
                  
                  <td>Morales</td>
                  
                  <td>293.38077</td>
                  
                  <td>free</td>
                  
                  <td>Corpus Christi, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532063507000</td>
                  
                  <td>563</td>
                  
                  <td>All In (feat. Gift of Gab &amp; Pigeon John)</td>
                  
                  <td>200</td>
                  
                  <td>1538426031000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>293</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 20:33:51</td>
                  
                  <td>20</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Carlos Vives</td>
                  
                  <td>Logged In</td>
                  
                  <td>Joseph</td>
                  
                  <td>M</td>
                  
                  <td>16</td>
                  
                  <td>Morales</td>
                  
                  <td>249.18159</td>
                  
                  <td>free</td>
                  
                  <td>Corpus Christi, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532063507000</td>
                  
                  <td>563</td>
                  
                  <td>Amor Latino</td>
                  
                  <td>200</td>
                  
                  <td>1538426324000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>293</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 20:38:44</td>
                  
                  <td>20</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Extreme</td>
                  
                  <td>Logged In</td>
                  
                  <td>Viviana</td>
                  
                  <td>F</td>
                  
                  <td>14</td>
                  
                  <td>Finley</td>
                  
                  <td>408.68526</td>
                  
                  <td>paid</td>
                  
                  <td>Gallup, NM</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1523777521000</td>
                  
                  <td>454</td>
                  
                  <td>Decadence Dance</td>
                  
                  <td>200</td>
                  
                  <td>1538428751000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>198</td>
                  
                  <td>0</td>
                  
                  <td>NM</td>
                  
                  <td>2018-10-01 21:19:11</td>
                  
                  <td>21</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Arianna</td>
                  
                  <td>F</td>
                  
                  <td>14</td>
                  
                  <td>Bullock</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Topeka, KS</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1538314334000</td>
                  
                  <td>282</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538430658000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>283</td>
                  
                  <td>0</td>
                  
                  <td>KS</td>
                  
                  <td>2018-10-01 21:50:58</td>
                  
                  <td>21</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Mystikal</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jaleel</td>
                  
                  <td>M</td>
                  
                  <td>38</td>
                  
                  <td>Maldonado</td>
                  
                  <td>360.64608</td>
                  
                  <td>free</td>
                  
                  <td>Boulder, CO</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537890437000</td>
                  
                  <td>407</td>
                  
                  <td>Not That Nigga</td>
                  
                  <td>200</td>
                  
                  <td>1538431689000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>59</td>
                  
                  <td>0</td>
                  
                  <td>CO</td>
                  
                  <td>2018-10-01 22:08:09</td>
                  
                  <td>22</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>POCAHONTAS</td>
                  
                  <td>Logged In</td>
                  
                  <td>Payton</td>
                  
                  <td>F</td>
                  
                  <td>4</td>
                  
                  <td>Campbell</td>
                  
                  <td>162.48118</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529027541000</td>
                  
                  <td>564</td>
                  
                  <td>Where Do I Go From Here</td>
                  
                  <td>200</td>
                  
                  <td>1538431705000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"</td>
                  
                  <td>39</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 22:08:25</td>
                  
                  <td>22</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Phish</td>
                  
                  <td>Logged In</td>
                  
                  <td>Viviana</td>
                  
                  <td>F</td>
                  
                  <td>28</td>
                  
                  <td>Finley</td>
                  
                  <td>204.72118</td>
                  
                  <td>paid</td>
                  
                  <td>Gallup, NM</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1523777521000</td>
                  
                  <td>454</td>
                  
                  <td>Twist (LP Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538432236000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>198</td>
                  
                  <td>0</td>
                  
                  <td>NM</td>
                  
                  <td>2018-10-01 22:17:16</td>
                  
                  <td>22</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Duckworth Lewis Method</td>
                  
                  <td>Logged In</td>
                  
                  <td>Erick</td>
                  
                  <td>M</td>
                  
                  <td>31</td>
                  
                  <td>Brooks</td>
                  
                  <td>258.16771</td>
                  
                  <td>free</td>
                  
                  <td>Selma, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537956751000</td>
                  
                  <td>57</td>
                  
                  <td>Flatten the Hay</td>
                  
                  <td>200</td>
                  
                  <td>1538432250000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>58</td>
                  
                  <td>0</td>
                  
                  <td>AL</td>
                  
                  <td>2018-10-01 22:17:30</td>
                  
                  <td>22</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Everett</td>
                  
                  <td>M</td>
                  
                  <td>79</td>
                  
                  <td>Quinn</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Appleton, WI</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1536082261000</td>
                  
                  <td>553</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538434233000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>195</td>
                  
                  <td>0</td>
                  
                  <td>WI</td>
                  
                  <td>2018-10-01 22:50:33</td>
                  
                  <td>22</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ean</td>
                  
                  <td>M</td>
                  
                  <td>5</td>
                  
                  <td>White</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Show Low, AZ</td>
                  
                  <td>GET</td>
                  
                  <td>Downgrade</td>
                  
                  <td>1535231759000</td>
                  
                  <td>593</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538434350000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>214</td>
                  
                  <td>0</td>
                  
                  <td>AZ</td>
                  
                  <td>2018-10-01 22:52:30</td>
                  
                  <td>22</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Lady GaGa / Colby O'Donis</td>
                  
                  <td>Logged In</td>
                  
                  <td>Payton</td>
                  
                  <td>F</td>
                  
                  <td>24</td>
                  
                  <td>Campbell</td>
                  
                  <td>238.54975</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529027541000</td>
                  
                  <td>564</td>
                  
                  <td>Just Dance</td>
                  
                  <td>200</td>
                  
                  <td>1538435080000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"</td>
                  
                  <td>39</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 23:04:40</td>
                  
                  <td>23</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Metallica</td>
                  
                  <td>Logged In</td>
                  
                  <td>Brayden</td>
                  
                  <td>M</td>
                  
                  <td>50</td>
                  
                  <td>Thomas</td>
                  
                  <td>450.19383</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534133898000</td>
                  
                  <td>556</td>
                  
                  <td>Ride The Lightning</td>
                  
                  <td>200</td>
                  
                  <td>1538435426000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>85</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 23:10:26</td>
                  
                  <td>23</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Cheetah Girls</td>
                  
                  <td>Logged In</td>
                  
                  <td>Payton</td>
                  
                  <td>F</td>
                  
                  <td>26</td>
                  
                  <td>Campbell</td>
                  
                  <td>183.24853</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529027541000</td>
                  
                  <td>564</td>
                  
                  <td>Shake A Tail Feather</td>
                  
                  <td>200</td>
                  
                  <td>1538435483000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"</td>
                  
                  <td>39</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 23:11:23</td>
                  
                  <td>23</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Lupe Fiasco</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lakyla</td>
                  
                  <td>F</td>
                  
                  <td>148</td>
                  
                  <td>Porter</td>
                  
                  <td>287.03302</td>
                  
                  <td>paid</td>
                  
                  <td>Modesto, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535849930000</td>
                  
                  <td>507</td>
                  
                  <td>Pressure [feat. Jay-Z] (Explicit Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538435542000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>162</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 23:12:22</td>
                  
                  <td>23</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Nomads</td>
                  
                  <td>Logged In</td>
                  
                  <td>Diego</td>
                  
                  <td>M</td>
                  
                  <td>10</td>
                  
                  <td>Mckee</td>
                  
                  <td>229.3024</td>
                  
                  <td>free</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537167593000</td>
                  
                  <td>444</td>
                  
                  <td>Showing Pictures To The Blind</td>
                  
                  <td>200</td>
                  
                  <td>1538437709000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>32</td>
                  
                  <td>0</td>
                  
                  <td>AZ</td>
                  
                  <td>2018-10-01 23:48:29</td>
                  
                  <td>23</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Disturbed</td>
                  
                  <td>Logged In</td>
                  
                  <td>Aurora</td>
                  
                  <td>F</td>
                  
                  <td>49</td>
                  
                  <td>Humphrey</td>
                  
                  <td>204.77342</td>
                  
                  <td>paid</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536795126000</td>
                  
                  <td>537</td>
                  
                  <td>Decadence (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538438125000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>127</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 23:55:25</td>
                  
                  <td>23</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Sanctus Real</td>
                  
                  <td>Logged In</td>
                  
                  <td>Yehoshua</td>
                  
                  <td>M</td>
                  
                  <td>4</td>
                  
                  <td>Maynard</td>
                  
                  <td>246.59546</td>
                  
                  <td>free</td>
                  
                  <td>Killeen-Temple, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1531747953000</td>
                  
                  <td>290</td>
                  
                  <td>The Way The World Turns</td>
                  
                  <td>200</td>
                  
                  <td>1538438976000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>291</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-02 00:09:36</td>
                  
                  <td>0</td>
                  
                  <td>2018-10-02 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Firehouse</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>260</td>
                  
                  <td>Myers</td>
                  
                  <td>244.74077</td>
                  
                  <td>paid</td>
                  
                  <td>Grand Rapids-Wyoming, MI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529995579000</td>
                  
                  <td>235</td>
                  
                  <td>When I Look Into Your Eyes</td>
                  
                  <td>200</td>
                  
                  <td>1538441924000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>236</td>
                  
                  <td>0</td>
                  
                  <td>MI</td>
                  
                  <td>2018-10-02 00:58:44</td>
                  
                  <td>0</td>
                  
                  <td>2018-10-02 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Blind Guardian</td>
                  
                  <td>Logged In</td>
                  
                  <td>Aurora</td>
                  
                  <td>F</td>
                  
                  <td>77</td>
                  
                  <td>Humphrey</td>
                  
                  <td>209.26649</td>
                  
                  <td>paid</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536795126000</td>
                  
                  <td>537</td>
                  
                  <td>Beyond the Ice (Remastered)</td>
                  
                  <td>200</td>
                  
                  <td>1538442683000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>127</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-02 01:11:23</td>
                  
                  <td>1</td>
                  
                  <td>2018-10-02 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jaleel</td>
                  
                  <td>M</td>
                  
                  <td>84</td>
                  
                  <td>Maldonado</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Boulder, CO</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1537890437000</td>
                  
                  <td>407</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538442718000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>59</td>
                  
                  <td>0</td>
                  
                  <td>CO</td>
                  
                  <td>2018-10-02 01:11:58</td>
                  
                  <td>1</td>
                  
                  <td>2018-10-02 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>26</td>
                  
                  <td>Raymond</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Hartford-West Hartford-East Hartford, CT</td>
                  
                  <td>GET</td>
                  
                  <td>About</td>
                  
                  <td>1534245996000</td>
                  
                  <td>574</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538443225000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                  <td>0</td>
                  
                  <td>CT</td>
                  
                  <td>2018-10-02 01:20:25</td>
                  
                  <td>1</td>
                  
                  <td>2018-10-02 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lillian</td>
                  
                  <td>F</td>
                  
                  <td>83</td>
                  
                  <td>Cameron</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Columbus, OH</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Down</td>
                  
                  <td>1533472700000</td>
                  
                  <td>471</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538449734000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>231</td>
                  
                  <td>0</td>
                  
                  <td>OH</td>
                  
                  <td>2018-10-02 03:08:54</td>
                  
                  <td>3</td>
                  
                  <td>2018-10-02 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                </tr>
                
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
  
  <script class="pd_save">
    $(function() {
      var tableWrapper = $('.df-table-wrapper-c58ef392');
      var fixedHeader = $('.fixed-header', tableWrapper);
      var tableContainer = $('.df-table-container', tableWrapper);
      var table = $('.df-table', tableContainer);
      var rows = $('tbody > tr', table);
      var total = 100;
  
      fixedHeader
        .css('width', table.width())
        .find('.fixed-cell')
        .each(function(i, e) {
          $(this).css('width', $('.df-table-wrapper-c58ef392 th:nth-child(' + (i+1) + ')').css('width'));
        });
  
      tableContainer.scroll(function() {
        fixedHeader.css({ left: table.position().left });
      });
  
      rows.on("click", function(e){
          var txt = e.delegateTarget.innerText;
          var splits = txt.split("\t");
          var len = splits.length;
          var hdrs = $(fixedHeader).find(".fixed-cell");
          // Add all cells in the selected row as a map to be consumed by the target as needed
          var payload = {type:"select", targetDivId: "" };
          for (var i = 0; i < len; i++) {
            payload[hdrs[i].innerHTML] = splits[i];
          }
  
          //simple selection highlighting, client adds "selected" class
          $(this).addClass("selected").siblings().removeClass("selected");
          $(document).trigger('pd_event', payload);
      });
  
      $('.df-table-search', tableWrapper).keyup(function() {
        var val = '^(?=.*\\b' + $.trim($(this).val()).split(/\s+/).join('\\b)(?=.*\\b') + ').*$';
        var reg = RegExp(val, 'i');
        var index = 0;
        
        rows.each(function(i, e) {
          if (!reg.test($(this).text().replace(/\s+/g, ' '))) {
            $(this).attr('class', 'hidden');
          }
          else {
            $(this).attr('class', (++index % 2 == 0 ? 'even' : 'odd'));
          }
        });
        $('.df-table-search-count', tableWrapper).html('Showing ' + index + ' of ' + total + ' rows');
      });
    });
  
    $(".df-table-wrapper td:contains('http://')").each(function(){var tc = this.textContent; $(this).wrapInner("<a target='_blank' href='" + tc + "'></a>");});
    $(".df-table-wrapper td:contains('https://')").each(function(){var tc = this.textContent; $(this).wrapInner("<a target='_blank' href='" + tc + "'></a>");});
  </script>
  
        </div>



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
      <td>543705</td>
      <td>528005</td>
      <td>543705</td>
      <td>543705</td>
      <td>543705</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mean</td>
      <td>107.30629109535502</td>
      <td>1.535523414862437E12</td>
      <td>2040.8143533717732</td>
      <td>210.01829116892432</td>
      <td>1.5409645412098003E12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stddev</td>
      <td>116.72350849188074</td>
      <td>3.0787254929957166E9</td>
      <td>1434.338931078271</td>
      <td>31.471919021567537</td>
      <td>1.4820571449105084E9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>min</td>
      <td>0</td>
      <td>1509854193000</td>
      <td>1</td>
      <td>200</td>
      <td>1538352011000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>max</td>
      <td>1005</td>
      <td>1543073874000</td>
      <td>4808</td>
      <td>404</td>
      <td>1543622466000</td>
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
    |                   21247|
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
    |                        345|
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
    |                       275|
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
    |                       192|
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
    |                 80292|
    +----------------------+
    
    None
    +---------------------------+
    |userAgent : distinct values|
    +---------------------------+
    |                         71|
    +---------------------------+
    
    None
    +------------------------+
    |userId : distinct values|
    +------------------------+
    |                     449|
    +------------------------+
    
    None
    


```python
# further analysis of relationship between page and status...
user_log.groupBy(["status", "method", "page"]).count().orderBy("status", "method", "count").show()
```

    +------+------+--------------------+------+
    |status|method|                page| count|
    +------+------+--------------------+------+
    |   200|   GET|            Register|    11|
    |   200|   GET|Cancellation Conf...|    99|
    |   200|   GET|             Upgrade|   968|
    |   200|   GET|               About|  1855|
    |   200|   GET|            Settings|  2964|
    |   200|   GET|                Help|  3150|
    |   200|   GET|           Downgrade|  3811|
    |   200|   GET|         Roll Advert|  7773|
    |   200|   GET|                Home| 27412|
    |   200|   PUT|     Add to Playlist| 12349|
    |   200|   PUT|            NextSong|432877|
    |   307|   PUT| Submit Registration|     4|
    |   307|   PUT|              Cancel|    99|
    |   307|   PUT|    Submit Downgrade|   117|
    |   307|   PUT|      Submit Upgrade|   287|
    |   307|   PUT|       Save Settings|   585|
    |   307|   PUT|         Thumbs Down|  4911|
    |   307|   PUT|              Logout|  5990|
    |   307|   PUT|               Login|  6011|
    |   307|   PUT|          Add Friend|  8087|
    +------+------+--------------------+------+
    only showing top 20 rows
    
    


```python
# detailled analysis of feature "userAgent"
pd_df = user_log.groupBy("userAgent").count().orderBy("count").toPandas()
print(pd_df["userAgent"].tolist())
pd_df["count"].describe()
```

    ['"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14"', 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:30.0) Gecko/20100101 Firefox/30.0', '"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.0; rv:31.0) Gecko/20100101 Firefox/31.0', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:30.0) Gecko/20100101 Firefox/30.0', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:32.0) Gecko/20100101 Firefox/32.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.74.9 (KHTML, like Gecko) Version/7.0.2 Safari/537.74.9"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', 'Mozilla/5.0 (Windows NT 5.1; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.76.4 (KHTML, like Gecko) Version/7.0.4 Safari/537.76.4"', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (iPad; CPU OS 7_1_1 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D201 Safari/9537.53"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/534.59.10 (KHTML, like Gecko) Version/5.1.9 Safari/534.59.10"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.3 (KHTML, like Gecko) Version/8.0 Safari/600.1.3"', '"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_1 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D201 Safari/9537.53"', 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36"', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D167 Safari/9537.53"', '"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/34.0.1847.116 Chrome/34.0.1847.116 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36"', 'Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/36.0.1985.125 Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/6.1.5 Safari/537.77.4"', '"Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"', 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; WOW64; Trident/6.0)', '"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.1; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.8 (KHTML, like Gecko) Version/8.0 Safari/600.1.8"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/6.1.5 Safari/537.77.4"', 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko', '"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0', 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0', 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko', 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"', '"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"', 'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (iPad; CPU OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"', None, 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.78.2 (KHTML, like Gecko) Version/7.0.6 Safari/537.78.2"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"', '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"']
    




    count       72.000000
    mean      7551.458333
    std       9805.527044
    min        245.000000
    25%       1802.250000
    50%       3546.000000
    75%       9839.500000
    max      46082.000000
    Name: count, dtype: float64




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

    null values in rows with auth value Logged Out:
    column artist : 15606
    column firstName : 15606
    column gender : 15606
    column lastName : 15606
    column length : 15606
    column location : 15606
    column registration : 15606
    column song : 15606
    column userAgent : 15606
    column userId : 0
    null values in rows with auth value Cancelled:
    column artist : 99
    column firstName : 0
    column gender : 0
    column lastName : 0
    column length : 99
    column location : 0
    column registration : 0
    column song : 99
    column userAgent : 0
    column userId : 0
    null values in rows with auth value Guest:
    column artist : 94
    column firstName : 94
    column gender : 94
    column lastName : 94
    column length : 94
    column location : 94
    column registration : 94
    column song : 94
    column userAgent : 94
    column userId : 0
    null values in rows with auth value Logged In:
    column artist : 95029
    column firstName : 0
    column gender : 0
    column lastName : 0
    column length : 95029
    column location : 0
    column registration : 0
    column song : 95029
    column userAgent : 0
    column userId : 0
    

## Remove values where "auth" value is either "Guest" or "Logged Out" since userId is missing


```python
print("Rows in dataset before auth values Guest and Logged Out are removed: ", user_log.count())

# remove values where "auth" value is either "Guest" or "Logged Out" since userId is missing
user_log = user_log.filter(user_log["auth"].isin(*["Guest", "Logged Out"]) == False)

print("Rows in dataset before auth values Guest and Logged Out are removed: ", user_log.count())
```

    Rows in dataset before auth values Guest and Logged Out are removed:  543705
    Rows in dataset before auth values Guest and Logged Out are removed:  528005
    

# Exploratory Data Analysis
## Define feature "churn"


```python
flag_churn_event = F.udf(lambda page: 1 if page == "Cancellation Confirmation" else 0, IntegerType())
user_log = user_log.withColumn("churn", flag_churn_event("page"))

# analyze churn per userId
pd_df = user_log.select("userId", "churn").dropDuplicates().toPandas()
print("Distinct users in dataset: {}".format(pd_df["userId"].nunique()))
print("Value counts of churn over all userId's:")
print(pd_df["churn"].value_counts())
plt.hist(pd_df["churn"])
plt.show()
```

    Distinct users in dataset: 448
    Value counts of churn over all userId's:
    0    448
    1     99
    Name: churn, dtype: int64
    


![png](output_18_1.png)


## Explore Data

### "location": analyse user locations and CSA (Combined Statistical Areas)


```python
# create new feature CSA (Combined Statistical Areas) from location 
get_csa = F.split(user_log["location"], ", ")
user_log = user_log.withColumn("CSA", get_csa.getItem(1))

# analyze CSA
pd_user_csa_churn = user_log.select("userId", "CSA", "churn").dropDuplicates().groupBy("userId", "CSA").agg(F.sum("churn").alias("churn")).toPandas()

pd_csa_analysis = pd_user_csa_churn.groupby("CSA", as_index=False).agg({"userId":"count", "churn": "sum"}).sort_values(by=["userId", "churn"], ascending = False)

plt.figure(figsize=(16, 16))
sns.barplot(x="userId", y="CSA" , data=pd_csa_analysis, color = "red");
sns.barplot(x="churn", y="CSA" , data=pd_csa_analysis, color = "blue").set_title("userId count (red) & churn count (blue) per CSA");
```


![png](output_20_0.png)


### Convert to timestamp "ts" time/date


```python
def convert_ts_to_datetime(df, column):
    get_datetime = F.udf(lambda timestamp: datetime.datetime.fromtimestamp(timestamp/1000).isoformat())
    df = df.withColumn(column + "_ts", get_datetime(df[column]).cast(TimestampType()))
    return df

# create new feature with Spark Timestamp data type
user_log = convert_ts_to_datetime(user_log, "ts")
```


```python
# analyze min and max timestamp data in set
min_date, max_date = user_log.select(F.min("ts_ts"), F.max("ts_ts")).first()
print("Minimum and Maximum timestamp data:")
min_date, max_date
```

    Minimum and Maximum timestamp data:
    




    (datetime.datetime(2018, 10, 1, 0, 0, 11),
     datetime.datetime(2018, 12, 1, 0, 1, 6))




```python
# get new features day and hour from ts
user_log = user_log.withColumn("ts_hour", F.hour("ts_ts"))
user_log = user_log.withColumn("ts_date", F.to_date("ts_ts"))

print("Analyze log data over time:")
#pd_df = user_log.select(hour("ts_ts").alias("hour")).groupBy("hour").count().orderBy("hour").toPandas()
pd_df = user_log.select("ts_hour").groupBy("ts_hour").count().orderBy("ts_hour").toPandas()
pd_df.plot.line(x="ts_hour", y="count");
```

    Analyze log data over time:
    


![png](output_24_1.png)


### Features from "page" value


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

# dictionary for page values and corresponding new features
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

## Create new dataframe for features per userId


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

## Encode binary features "gender", "level"

### Encode "gender"
- gender value "M" = value 1
- gender value "F" = value 0

### Encode "level"
- level value "paid" = value 1
- level value "free" = value 0


```python
# one hot encode gender in original df
one_hot_encode_gender = F.udf(lambda gender: 1 if gender == "M" else 0, IntegerType())
user_log = user_log.withColumn("gender_bin", one_hot_encode_gender("gender"))

# join binary gender on userId in features df
user_gender_selection =  user_log.select(["userId", "gender_bin"]).dropDuplicates(subset=['userId'])
features_df = features_df.join(user_gender_selection, "userId")

# one hot encode level in original df
one_hot_encode_level = F.udf(lambda level: 1 if level == "paid" else 0, IntegerType())
user_log = user_log.withColumn("level_bin", one_hot_encode_level("level"))

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


<style type="text/css">.pd_warning{display:none;}</style><div class="pd_warning"><em>Hey, there's something awesome here! To see it, open this notebook outside GitHub, in a viewer like Jupyter</em></div>
        <div class="pd_save is-viewer-good" style="padding-right:10px;text-align: center;line-height:initial !important;font-size: xx-large;font-weight: 500;color: coral;">
            
        </div>
    <div id="chartFigureb3631c42" class="pd_save is-viewer-good" style="overflow-x:auto">
            <style type="text/css" class="pd_save">
    .df-table-wrapper .panel-heading {
      border-radius: 0;
      padding: 0px;
    }
    .df-table-wrapper .panel-heading:hover {
      border-color: #008571;
    }
    .df-table-wrapper .panel-title a {
      background-color: #f9f9fb;
      color: #333333;
      display: block;
      outline: none;
      padding: 10px 15px;
      text-decoration: none;
    }
    .df-table-wrapper .panel-title a:hover {
      background-color: #337ab7;
      border-color: #2e6da4;
      color: #ffffff;
      display: block;
      padding: 10px 15px;
      text-decoration: none;
    }
    .df-table-wrapper {
      font-size: small;
      font-weight: 300;
      letter-spacing: 0.5px;
      line-height: normal;
      height: inherit;
      overflow: auto;
    }
    .df-table-search {
      margin: 0 0 20px 0;
    }
    .df-table-search-count {
      display: inline-block;
      margin: 0 0 20px 0;
    }
    .df-table-container {
      max-height: 50vh;
      max-width: 100%;
      overflow-x: auto;
      position: relative;
    }
    .df-table-wrapper table {
      border: 0 none #ffffff;
      border-collapse: collapse;
      margin: 0;
      min-width: 100%;
      padding: 0;
      table-layout: fixed;
      height: inherit;
      overflow: auto;
    }
    .df-table-wrapper tr.hidden {
      display: none;
    }
    .df-table-wrapper tr:nth-child(even) {
      background-color: #f9f9fb;
    }
    .df-table-wrapper tr.even {
      background-color: #f9f9fb;
    }
    .df-table-wrapper tr.odd {
      background-color: #ffffff;
    }
    .df-table-wrapper td + td {
      border-left: 1px solid #e0e0e0;
    }
  
    .df-table-wrapper thead,
    .fixed-header {
      font-weight: 600;
    }
    .df-table-wrapper tr,
    .fixed-row {
      border: 0 none #ffffff;
      margin: 0;
      padding: 0;
    }
    .df-table-wrapper th,
    .df-table-wrapper td,
    .fixed-cell {
      border: 0 none #ffffff;
      margin: 0;
      min-width: 50px;
      padding: 5px 20px 5px 10px;
      text-align: left;
      word-wrap: break-word;
    }
    .df-table-wrapper th {
      padding-bottom: 0;
      padding-top: 0;
    }
    .df-table-wrapper th div {
      max-height: 1px;
      visibility: hidden;
    }
  
    .df-schema-field {
      margin-left: 10px;
    }
  
    .fixed-header-container {
      overflow: hidden;
      position: relative;
    }
    .fixed-header {
      border-bottom: 2px solid #000;
      display: table;
      position: relative;
    }
    .fixed-row {
      display: table-row;
    }
    .fixed-cell {
      display: table-cell;
    }
  </style>
  
  
  <div class="df-table-wrapper df-table-wrapper-b3631c42 panel-group pd_save">
    <!-- dataframe schema -->
    
    <div class="panel panel-default">
      <div class="panel-heading">
        <h4 class="panel-title" style="margin: 0px;">
          <a data-toggle="collapse" href="#df-schema-b3631c42" data-parent="#df-table-wrapper-b3631c42">Schema</a>
        </h4>
      </div>
      <div id="df-schema-b3631c42" class="panel-collapse collapse">
        <div class="panel-body" style="font-family: monospace;">
          <div class="df-schema-fields">
            <div>Field types:</div>
            
              <div class="df-schema-field"><strong>userId: </strong> int32</div>
            
              <div class="df-schema-field"><strong>label: </strong> int32</div>
            
              <div class="df-schema-field"><strong>gender_bin: </strong> int32</div>
            
              <div class="df-schema-field"><strong>level_bin: </strong> int32</div>
            
              <div class="df-schema-field"><strong>sum(downgraded): </strong> int64</div>
            
              <div class="df-schema-field"><strong>sum(upgraded): </strong> int64</div>
            
              <div class="df-schema-field"><strong>sum(advert_shown): </strong> int64</div>
            
              <div class="df-schema-field"><strong>sum(thumps_down): </strong> int64</div>
            
              <div class="df-schema-field"><strong>sum(thumps_up): </strong> int64</div>
            
              <div class="df-schema-field"><strong>sum(friend_added): </strong> int64</div>
            
              <div class="df-schema-field"><strong>sum(song_added): </strong> int64</div>
            
              <div class="df-schema-field"><strong>song_count: </strong> int64</div>
            
              <div class="df-schema-field"><strong>days_since_reg: </strong> int32</div>
            
              <div class="df-schema-field"><strong>total_session_time_sec: </strong> int64</div>
            
              <div class="df-schema-field"><strong>sum(downgraded)_scaled: </strong> float64</div>
            
              <div class="df-schema-field"><strong>sum(upgraded)_scaled: </strong> float64</div>
            
              <div class="df-schema-field"><strong>sum(advert_shown)_scaled: </strong> float64</div>
            
              <div class="df-schema-field"><strong>sum(thumps_down)_scaled: </strong> float64</div>
            
              <div class="df-schema-field"><strong>sum(thumps_up)_scaled: </strong> float64</div>
            
              <div class="df-schema-field"><strong>sum(friend_added)_scaled: </strong> float64</div>
            
              <div class="df-schema-field"><strong>sum(song_added)_scaled: </strong> float64</div>
            
              <div class="df-schema-field"><strong>song_count_scaled: </strong> float64</div>
            
              <div class="df-schema-field"><strong>days_since_reg_scaled: </strong> float64</div>
            
              <div class="df-schema-field"><strong>total_session_time_sec_scaled: </strong> float64</div>
            
              <div class="df-schema-field"><strong>features: </strong> object</div>
            
              <div class="df-schema-field"><strong>pcaFeatures: </strong> object</div>
            
          </div>
        </div>
      </div>
    </div>
    
    <!-- dataframe table -->
    <div class="panel panel-default">
      
      <div class="panel-heading">
        <h4 class="panel-title" style="margin: 0px;">
          <a data-toggle="collapse" href="#df-table-b3631c42" data-parent="#df-table-wrapper-b3631c42"> Table</a>
        </h4>
      </div>
      
      <div id="df-table-b3631c42" class="panel-collapse collapse in">
        <div class="panel-body">
          
          <input type="text" class="df-table-search form-control input-sm" placeholder="Search table">
          
          <div>
            
            <span class="df-table-search-count">Showing 100 of 448 rows</span>
            
          </div>
          <!-- fixed header for when dataframe table scrolls -->
          <div class="fixed-header-container">
            <div class="fixed-header">
              <div class="fixed-row">
                
                <div class="fixed-cell">userId</div>
                
                <div class="fixed-cell">label</div>
                
                <div class="fixed-cell">gender_bin</div>
                
                <div class="fixed-cell">level_bin</div>
                
                <div class="fixed-cell">sum(downgraded)</div>
                
                <div class="fixed-cell">sum(upgraded)</div>
                
                <div class="fixed-cell">sum(advert_shown)</div>
                
                <div class="fixed-cell">sum(thumps_down)</div>
                
                <div class="fixed-cell">sum(thumps_up)</div>
                
                <div class="fixed-cell">sum(friend_added)</div>
                
                <div class="fixed-cell">sum(song_added)</div>
                
                <div class="fixed-cell">song_count</div>
                
                <div class="fixed-cell">days_since_reg</div>
                
                <div class="fixed-cell">total_session_time_sec</div>
                
                <div class="fixed-cell">sum(downgraded)_scaled</div>
                
                <div class="fixed-cell">sum(upgraded)_scaled</div>
                
                <div class="fixed-cell">sum(advert_shown)_scaled</div>
                
                <div class="fixed-cell">sum(thumps_down)_scaled</div>
                
                <div class="fixed-cell">sum(thumps_up)_scaled</div>
                
                <div class="fixed-cell">sum(friend_added)_scaled</div>
                
                <div class="fixed-cell">sum(song_added)_scaled</div>
                
                <div class="fixed-cell">song_count_scaled</div>
                
                <div class="fixed-cell">days_since_reg_scaled</div>
                
                <div class="fixed-cell">total_session_time_sec_scaled</div>
                
                <div class="fixed-cell">features</div>
                
                <div class="fixed-cell">pcaFeatures</div>
                
              </div>
            </div>
          </div>
          <div class="df-table-container">
            <table class="df-table">
              <thead>
                <tr>
                  
                  <th><div>userId</div></th>
                  
                  <th><div>label</div></th>
                  
                  <th><div>gender_bin</div></th>
                  
                  <th><div>level_bin</div></th>
                  
                  <th><div>sum(downgraded)</div></th>
                  
                  <th><div>sum(upgraded)</div></th>
                  
                  <th><div>sum(advert_shown)</div></th>
                  
                  <th><div>sum(thumps_down)</div></th>
                  
                  <th><div>sum(thumps_up)</div></th>
                  
                  <th><div>sum(friend_added)</div></th>
                  
                  <th><div>sum(song_added)</div></th>
                  
                  <th><div>song_count</div></th>
                  
                  <th><div>days_since_reg</div></th>
                  
                  <th><div>total_session_time_sec</div></th>
                  
                  <th><div>sum(downgraded)_scaled</div></th>
                  
                  <th><div>sum(upgraded)_scaled</div></th>
                  
                  <th><div>sum(advert_shown)_scaled</div></th>
                  
                  <th><div>sum(thumps_down)_scaled</div></th>
                  
                  <th><div>sum(thumps_up)_scaled</div></th>
                  
                  <th><div>sum(friend_added)_scaled</div></th>
                  
                  <th><div>sum(song_added)_scaled</div></th>
                  
                  <th><div>song_count_scaled</div></th>
                  
                  <th><div>days_since_reg_scaled</div></th>
                  
                  <th><div>total_session_time_sec_scaled</div></th>
                  
                  <th><div>features</div></th>
                  
                  <th><div>pcaFeatures</div></th>
                  
                </tr>
              </thead>
              <tbody>
                
                <tr>
                  
                  <td>300042</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>12</td>
                  
                  <td>19</td>
                  
                  <td>121</td>
                  
                  <td>19</td>
                  
                  <td>41</td>
                  
                  <td>1397</td>
                  
                  <td>62</td>
                  
                  <td>346937</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.064</td>
                  
                  <td>0.237</td>
                  
                  <td>0.239</td>
                  
                  <td>0.12</td>
                  
                  <td>0.165</td>
                  
                  <td>0.171</td>
                  
                  <td>0.159</td>
                  
                  <td>0.173</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.064,0.237,0.239,0.12,0.165,0.171,0.159,0.173]</td>
                  
                  <td>[1.1434980383878257,1.002172659479925,-0.011041375975718894]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200049</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>3</td>
                  
                  <td>3</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>26</td>
                  
                  <td>112</td>
                  
                  <td>7603</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.011</td>
                  
                  <td>0.037</td>
                  
                  <td>0.006</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.003</td>
                  
                  <td>0.287</td>
                  
                  <td>0.003</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.011,0.037,0.006,0.0,0.0,0.003,0.287,0.003]</td>
                  
                  <td>[0.010376465231058693,1.0035827648121824,-0.007990431152476073]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>254</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>3</td>
                  
                  <td>7</td>
                  
                  <td>55</td>
                  
                  <td>25</td>
                  
                  <td>28</td>
                  
                  <td>993</td>
                  
                  <td>76</td>
                  
                  <td>245005</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.016</td>
                  
                  <td>0.087</td>
                  
                  <td>0.109</td>
                  
                  <td>0.158</td>
                  
                  <td>0.113</td>
                  
                  <td>0.121</td>
                  
                  <td>0.195</td>
                  
                  <td>0.122</td>
                  
                  <td>[1.0,0.0,0.333,0.333,0.016,0.087,0.109,0.158,0.113,0.121,0.195,0.122]</td>
                  
                  <td>[0.20992136273212944,1.0137548348002454,0.4223043640459932]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>151</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>7</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>2</td>
                  
                  <td>55</td>
                  
                  <td>38</td>
                  
                  <td>13681</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.037</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.019</td>
                  
                  <td>0.008</td>
                  
                  <td>0.007</td>
                  
                  <td>0.097</td>
                  
                  <td>0.006</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.037,0.0,0.0,0.019,0.008,0.007,0.097,0.006]</td>
                  
                  <td>[0.0073693773465699376,1.0008531989592744,-0.015722092901527582]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>29</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>73</td>
                  
                  <td>17</td>
                  
                  <td>90</td>
                  
                  <td>40</td>
                  
                  <td>48</td>
                  
                  <td>1907</td>
                  
                  <td>76</td>
                  
                  <td>464709</td>
                  
                  <td>0.333</td>
                  
                  <td>0.667</td>
                  
                  <td>0.39</td>
                  
                  <td>0.212</td>
                  
                  <td>0.178</td>
                  
                  <td>0.253</td>
                  
                  <td>0.194</td>
                  
                  <td>0.233</td>
                  
                  <td>0.195</td>
                  
                  <td>0.232</td>
                  
                  <td>[1.0,1.0,0.333,0.667,0.39,0.212,0.178,0.253,0.194,0.233,0.195,0.232]</td>
                  
                  <td>[1.265386854021701,1.0102032679084865,0.3649950150310262]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>256</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>9</td>
                  
                  <td>31</td>
                  
                  <td>16</td>
                  
                  <td>17</td>
                  
                  <td>723</td>
                  
                  <td>56</td>
                  
                  <td>179939</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.113</td>
                  
                  <td>0.061</td>
                  
                  <td>0.101</td>
                  
                  <td>0.069</td>
                  
                  <td>0.088</td>
                  
                  <td>0.144</td>
                  
                  <td>0.09</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.0,0.113,0.061,0.101,0.069,0.088,0.144,0.09]</td>
                  
                  <td>[0.9715257509403865,-0.01079969981318056,-0.28295698141511677]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>99</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>27</td>
                  
                  <td>7</td>
                  
                  <td>46</td>
                  
                  <td>23</td>
                  
                  <td>22</td>
                  
                  <td>1012</td>
                  
                  <td>133</td>
                  
                  <td>244902</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.144</td>
                  
                  <td>0.087</td>
                  
                  <td>0.091</td>
                  
                  <td>0.146</td>
                  
                  <td>0.089</td>
                  
                  <td>0.124</td>
                  
                  <td>0.341</td>
                  
                  <td>0.122</td>
                  
                  <td>[0.0,0.0,0.333,0.333,0.144,0.087,0.091,0.146,0.089,0.124,0.341,0.122]</td>
                  
                  <td>[0.2012727593576384,0.01698930497798527,0.48059397312394797]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300013</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>50</td>
                  
                  <td>4</td>
                  
                  <td>21</td>
                  
                  <td>571</td>
                  
                  <td>94</td>
                  
                  <td>139682</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.005</td>
                  
                  <td>0.062</td>
                  
                  <td>0.099</td>
                  
                  <td>0.025</td>
                  
                  <td>0.085</td>
                  
                  <td>0.07</td>
                  
                  <td>0.241</td>
                  
                  <td>0.069</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.005,0.062,0.099,0.025,0.085,0.07,0.241,0.069]</td>
                  
                  <td>[0.9529124562335094,0.9870792079766189,-0.35103553144717403]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>242</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>5</td>
                  
                  <td>6</td>
                  
                  <td>109</td>
                  
                  <td>85</td>
                  
                  <td>25402</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.016</td>
                  
                  <td>0.013</td>
                  
                  <td>0.01</td>
                  
                  <td>0.032</td>
                  
                  <td>0.024</td>
                  
                  <td>0.013</td>
                  
                  <td>0.218</td>
                  
                  <td>0.012</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.016,0.013,0.01,0.032,0.024,0.013,0.218,0.012]</td>
                  
                  <td>[0.018732541703861537,1.0031726681940418,0.00295468886313951]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200015</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>26</td>
                  
                  <td>10</td>
                  
                  <td>20</td>
                  
                  <td>7</td>
                  
                  <td>9</td>
                  
                  <td>408</td>
                  
                  <td>107</td>
                  
                  <td>97352</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.139</td>
                  
                  <td>0.125</td>
                  
                  <td>0.04</td>
                  
                  <td>0.044</td>
                  
                  <td>0.036</td>
                  
                  <td>0.05</td>
                  
                  <td>0.274</td>
                  
                  <td>0.048</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.139,0.125,0.04,0.044,0.036,0.05,0.274,0.048]</td>
                  
                  <td>[0.059269190707718626,1.008788056579576,0.10347660399151862]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>184</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>21</td>
                  
                  <td>61</td>
                  
                  <td>18</td>
                  
                  <td>34</td>
                  
                  <td>1316</td>
                  
                  <td>141</td>
                  
                  <td>327086</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.263</td>
                  
                  <td>0.121</td>
                  
                  <td>0.114</td>
                  
                  <td>0.137</td>
                  
                  <td>0.161</td>
                  
                  <td>0.362</td>
                  
                  <td>0.163</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.0,0.263,0.121,0.114,0.137,0.161,0.362,0.163]</td>
                  
                  <td>[1.0467390591265051,-0.0011246664182156482,-0.14575351609819615]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>261</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>27</td>
                  
                  <td>3</td>
                  
                  <td>17</td>
                  
                  <td>3</td>
                  
                  <td>11</td>
                  
                  <td>388</td>
                  
                  <td>91</td>
                  
                  <td>92909</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.144</td>
                  
                  <td>0.037</td>
                  
                  <td>0.034</td>
                  
                  <td>0.019</td>
                  
                  <td>0.044</td>
                  
                  <td>0.047</td>
                  
                  <td>0.233</td>
                  
                  <td>0.046</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.144,0.037,0.034,0.019,0.044,0.047,0.233,0.046]</td>
                  
                  <td>[0.038858624348844975,1.0058540438556542,0.06517291666427702]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300017</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>25</td>
                  
                  <td>388</td>
                  
                  <td>98</td>
                  
                  <td>125</td>
                  
                  <td>4283</td>
                  
                  <td>75</td>
                  
                  <td>1054897</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.027</td>
                  
                  <td>0.312</td>
                  
                  <td>0.767</td>
                  
                  <td>0.62</td>
                  
                  <td>0.504</td>
                  
                  <td>0.524</td>
                  
                  <td>0.192</td>
                  
                  <td>0.527</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.027,0.312,0.767,0.62,0.504,0.524,0.192,0.527]</td>
                  
                  <td>[1.4227375355233944,0.0231770138202297,0.46320764978820217]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200004</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>98</td>
                  
                  <td>42</td>
                  
                  <td>80</td>
                  
                  <td>16</td>
                  
                  <td>40</td>
                  
                  <td>1430</td>
                  
                  <td>67</td>
                  
                  <td>348674</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.524</td>
                  
                  <td>0.525</td>
                  
                  <td>0.158</td>
                  
                  <td>0.101</td>
                  
                  <td>0.161</td>
                  
                  <td>0.175</td>
                  
                  <td>0.172</td>
                  
                  <td>0.174</td>
                  
                  <td>[1.0,0.0,0.0,0.333,0.524,0.525,0.158,0.101,0.161,0.175,0.172,0.174]</td>
                  
                  <td>[0.29316688638857824,1.031616800340238,0.5966375215114097]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>70</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>34</td>
                  
                  <td>138</td>
                  
                  <td>46</td>
                  
                  <td>84</td>
                  
                  <td>2914</td>
                  
                  <td>161</td>
                  
                  <td>720070</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.027</td>
                  
                  <td>0.425</td>
                  
                  <td>0.273</td>
                  
                  <td>0.291</td>
                  
                  <td>0.339</td>
                  
                  <td>0.356</td>
                  
                  <td>0.413</td>
                  
                  <td>0.36</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.027,0.425,0.273,0.291,0.339,0.356,0.413,0.36]</td>
                  
                  <td>[1.2286182473611824,1.0134159249251338,0.1255280684813082]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>203</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>26</td>
                  
                  <td>17</td>
                  
                  <td>90</td>
                  
                  <td>39</td>
                  
                  <td>49</td>
                  
                  <td>1682</td>
                  
                  <td>110</td>
                  
                  <td>424103</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.139</td>
                  
                  <td>0.212</td>
                  
                  <td>0.178</td>
                  
                  <td>0.247</td>
                  
                  <td>0.198</td>
                  
                  <td>0.206</td>
                  
                  <td>0.282</td>
                  
                  <td>0.212</td>
                  
                  <td>[1.0,0.0,0.333,0.333,0.139,0.212,0.178,0.247,0.198,0.206,0.282,0.212]</td>
                  
                  <td>[0.3008959535225072,1.023610343480766,0.6061805819298891]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>103</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>21</td>
                  
                  <td>2</td>
                  
                  <td>21</td>
                  
                  <td>9</td>
                  
                  <td>17</td>
                  
                  <td>423</td>
                  
                  <td>64</td>
                  
                  <td>101151</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.112</td>
                  
                  <td>0.025</td>
                  
                  <td>0.042</td>
                  
                  <td>0.057</td>
                  
                  <td>0.069</td>
                  
                  <td>0.052</td>
                  
                  <td>0.164</td>
                  
                  <td>0.05</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.112,0.025,0.042,0.057,0.069,0.052,0.164,0.05]</td>
                  
                  <td>[1.0069019531558625,-0.00789543252801103,-0.19306624068967684]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300007</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>10</td>
                  
                  <td>59</td>
                  
                  <td>12</td>
                  
                  <td>20</td>
                  
                  <td>751</td>
                  
                  <td>64</td>
                  
                  <td>182534</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.005</td>
                  
                  <td>0.125</td>
                  
                  <td>0.117</td>
                  
                  <td>0.076</td>
                  
                  <td>0.081</td>
                  
                  <td>0.092</td>
                  
                  <td>0.164</td>
                  
                  <td>0.091</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.005,0.125,0.117,0.076,0.081,0.092,0.164,0.091]</td>
                  
                  <td>[0.9817466399993053,0.9889130585720544,-0.3034812447670378]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>75</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>3</td>
                  
                  <td>10</td>
                  
                  <td>2</td>
                  
                  <td>6</td>
                  
                  <td>374</td>
                  
                  <td>52</td>
                  
                  <td>91104</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.005</td>
                  
                  <td>0.037</td>
                  
                  <td>0.02</td>
                  
                  <td>0.013</td>
                  
                  <td>0.024</td>
                  
                  <td>0.046</td>
                  
                  <td>0.133</td>
                  
                  <td>0.045</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.005,0.037,0.02,0.013,0.024,0.046,0.133,0.045]</td>
                  
                  <td>[0.9149790532649725,-0.015869831505382556,-0.3779723903960738]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>149</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>35</td>
                  
                  <td>11</td>
                  
                  <td>20</td>
                  
                  <td>492</td>
                  
                  <td>53</td>
                  
                  <td>124567</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.005</td>
                  
                  <td>0.025</td>
                  
                  <td>0.069</td>
                  
                  <td>0.07</td>
                  
                  <td>0.081</td>
                  
                  <td>0.06</td>
                  
                  <td>0.136</td>
                  
                  <td>0.062</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.005,0.025,0.069,0.07,0.081,0.06,0.136,0.062]</td>
                  
                  <td>[1.0196109655968388,0.9907015876155656,-0.2361469431365556]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>105</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>6</td>
                  
                  <td>35</td>
                  
                  <td>165</td>
                  
                  <td>87</td>
                  
                  <td>89</td>
                  
                  <td>2979</td>
                  
                  <td>78</td>
                  
                  <td>748601</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.032</td>
                  
                  <td>0.438</td>
                  
                  <td>0.326</td>
                  
                  <td>0.551</td>
                  
                  <td>0.359</td>
                  
                  <td>0.364</td>
                  
                  <td>0.2</td>
                  
                  <td>0.374</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.032,0.438,0.326,0.551,0.359,0.364,0.2,0.374]</td>
                  
                  <td>[1.3623749859243968,1.0211469214761768,0.34932457255950494]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100044</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>27</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>2</td>
                  
                  <td>1</td>
                  
                  <td>132</td>
                  
                  <td>4</td>
                  
                  <td>30561</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.144</td>
                  
                  <td>0.0</td>
                  
                  <td>0.008</td>
                  
                  <td>0.013</td>
                  
                  <td>0.004</td>
                  
                  <td>0.016</td>
                  
                  <td>0.01</td>
                  
                  <td>0.015</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.144,0.0,0.008,0.013,0.004,0.016,0.01,0.015]</td>
                  
                  <td>[0.009041483247779753,1.0005732359574138,0.007323503329141043]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>77</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>9</td>
                  
                  <td>4</td>
                  
                  <td>2</td>
                  
                  <td>154</td>
                  
                  <td>30</td>
                  
                  <td>37916</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.018</td>
                  
                  <td>0.025</td>
                  
                  <td>0.008</td>
                  
                  <td>0.019</td>
                  
                  <td>0.077</td>
                  
                  <td>0.019</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.0,0.0,0.018,0.025,0.008,0.019,0.077,0.019]</td>
                  
                  <td>[0.8981653287198007,-0.018225602861549395,-0.4101758959948131]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100038</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>86</td>
                  
                  <td>8</td>
                  
                  <td>26</td>
                  
                  <td>10</td>
                  
                  <td>12</td>
                  
                  <td>503</td>
                  
                  <td>34</td>
                  
                  <td>124231</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.46</td>
                  
                  <td>0.1</td>
                  
                  <td>0.051</td>
                  
                  <td>0.063</td>
                  
                  <td>0.048</td>
                  
                  <td>0.061</td>
                  
                  <td>0.087</td>
                  
                  <td>0.062</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.46,0.1,0.051,0.063,0.048,0.061,0.087,0.062]</td>
                  
                  <td>[0.06312414500212847,1.0086516341848109,0.17208657525029208]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>42</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>101</td>
                  
                  <td>41</td>
                  
                  <td>187</td>
                  
                  <td>71</td>
                  
                  <td>111</td>
                  
                  <td>4070</td>
                  
                  <td>67</td>
                  
                  <td>1077242</td>
                  
                  <td>0.333</td>
                  
                  <td>0.667</td>
                  
                  <td>0.54</td>
                  
                  <td>0.512</td>
                  
                  <td>0.37</td>
                  
                  <td>0.449</td>
                  
                  <td>0.448</td>
                  
                  <td>0.498</td>
                  
                  <td>0.172</td>
                  
                  <td>0.538</td>
                  
                  <td>[0.0,1.0,0.333,0.667,0.54,0.512,0.37,0.449,0.448,0.498,0.172,0.538]</td>
                  
                  <td>[1.51819057402181,0.03360874859523092,0.8611048022117993]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>197</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>92</td>
                  
                  <td>52</td>
                  
                  <td>21844</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.021</td>
                  
                  <td>0.0</td>
                  
                  <td>0.006</td>
                  
                  <td>0.0</td>
                  
                  <td>0.008</td>
                  
                  <td>0.011</td>
                  
                  <td>0.133</td>
                  
                  <td>0.011</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.021,0.0,0.006,0.0,0.008,0.011,0.133,0.011]</td>
                  
                  <td>[0.007028847018991902,1.0011074607657744,-0.01862475105518136]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100011</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>3</td>
                  
                  <td>42</td>
                  
                  <td>17</td>
                  
                  <td>33</td>
                  
                  <td>24</td>
                  
                  <td>24</td>
                  
                  <td>1021</td>
                  
                  <td>56</td>
                  
                  <td>250711</td>
                  
                  <td>0.667</td>
                  
                  <td>1.0</td>
                  
                  <td>0.225</td>
                  
                  <td>0.212</td>
                  
                  <td>0.065</td>
                  
                  <td>0.152</td>
                  
                  <td>0.097</td>
                  
                  <td>0.125</td>
                  
                  <td>0.144</td>
                  
                  <td>0.125</td>
                  
                  <td>[1.0,1.0,0.667,1.0,0.225,0.212,0.065,0.152,0.097,0.125,0.144,0.125]</td>
                  
                  <td>[1.2696303648002187,1.005255711979686,0.441623709724291]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300027</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>26</td>
                  
                  <td>5</td>
                  
                  <td>27</td>
                  
                  <td>4</td>
                  
                  <td>7</td>
                  
                  <td>269</td>
                  
                  <td>55</td>
                  
                  <td>62460</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.139</td>
                  
                  <td>0.062</td>
                  
                  <td>0.053</td>
                  
                  <td>0.025</td>
                  
                  <td>0.028</td>
                  
                  <td>0.033</td>
                  
                  <td>0.141</td>
                  
                  <td>0.031</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.139,0.062,0.053,0.025,0.028,0.033,0.141,0.031]</td>
                  
                  <td>[0.03880630834217596,0.006028160411677885,0.10130802868773764]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>89</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>28</td>
                  
                  <td>1</td>
                  
                  <td>17</td>
                  
                  <td>10</td>
                  
                  <td>9</td>
                  
                  <td>302</td>
                  
                  <td>86</td>
                  
                  <td>74888</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.15</td>
                  
                  <td>0.013</td>
                  
                  <td>0.034</td>
                  
                  <td>0.063</td>
                  
                  <td>0.036</td>
                  
                  <td>0.037</td>
                  
                  <td>0.221</td>
                  
                  <td>0.037</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.15,0.013,0.034,0.063,0.036,0.037,0.221,0.037]</td>
                  
                  <td>[0.037158601526185914,1.0054084998405337,0.06214787356454887]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100028</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>42</td>
                  
                  <td>7</td>
                  
                  <td>23</td>
                  
                  <td>10</td>
                  
                  <td>12</td>
                  
                  <td>552</td>
                  
                  <td>31</td>
                  
                  <td>135746</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.225</td>
                  
                  <td>0.087</td>
                  
                  <td>0.045</td>
                  
                  <td>0.063</td>
                  
                  <td>0.048</td>
                  
                  <td>0.067</td>
                  
                  <td>0.079</td>
                  
                  <td>0.067</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.225,0.087,0.045,0.063,0.048,0.067,0.079,0.067]</td>
                  
                  <td>[1.0203592473714087,-0.006549443802365469,-0.14604891818372248]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300030</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>42</td>
                  
                  <td>10</td>
                  
                  <td>13</td>
                  
                  <td>546</td>
                  
                  <td>55</td>
                  
                  <td>137352</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.062</td>
                  
                  <td>0.083</td>
                  
                  <td>0.063</td>
                  
                  <td>0.052</td>
                  
                  <td>0.067</td>
                  
                  <td>0.141</td>
                  
                  <td>0.068</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.0,0.062,0.083,0.063,0.052,0.067,0.141,0.068]</td>
                  
                  <td>[0.9496411174826309,-0.013179336353390214,-0.32150106731580397]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300035</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>16</td>
                  
                  <td>64</td>
                  
                  <td>498</td>
                  
                  <td>91</td>
                  
                  <td>186</td>
                  
                  <td>5528</td>
                  
                  <td>63</td>
                  
                  <td>1354984</td>
                  
                  <td>0.333</td>
                  
                  <td>0.667</td>
                  
                  <td>0.086</td>
                  
                  <td>0.8</td>
                  
                  <td>0.984</td>
                  
                  <td>0.576</td>
                  
                  <td>0.75</td>
                  
                  <td>0.676</td>
                  
                  <td>0.162</td>
                  
                  <td>0.677</td>
                  
                  <td>[0.0,1.0,0.333,0.667,0.086,0.8,0.984,0.576,0.75,0.676,0.162,0.677]</td>
                  
                  <td>[1.7923451414647569,0.05140106073230684,1.21802587059434]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>111</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>7</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>8</td>
                  
                  <td>1</td>
                  
                  <td>93</td>
                  
                  <td>61</td>
                  
                  <td>28078</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.037</td>
                  
                  <td>0.0</td>
                  
                  <td>0.01</td>
                  
                  <td>0.051</td>
                  
                  <td>0.004</td>
                  
                  <td>0.011</td>
                  
                  <td>0.156</td>
                  
                  <td>0.014</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.037,0.0,0.01,0.051,0.004,0.011,0.156,0.014]</td>
                  
                  <td>[0.015406697099632799,0.0033281614289569633,0.03853075149434917]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>144</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>9</td>
                  
                  <td>1</td>
                  
                  <td>6</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>157</td>
                  
                  <td>92</td>
                  
                  <td>38418</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.048</td>
                  
                  <td>0.013</td>
                  
                  <td>0.012</td>
                  
                  <td>0.013</td>
                  
                  <td>0.008</td>
                  
                  <td>0.019</td>
                  
                  <td>0.236</td>
                  
                  <td>0.019</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.048,0.013,0.012,0.013,0.008,0.019,0.236,0.019]</td>
                  
                  <td>[0.015500977852220866,1.0033120623164047,0.00528684112823537]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>145</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>1</td>
                  
                  <td>21</td>
                  
                  <td>4</td>
                  
                  <td>27</td>
                  
                  <td>10</td>
                  
                  <td>23</td>
                  
                  <td>594</td>
                  
                  <td>100</td>
                  
                  <td>144462</td>
                  
                  <td>0.667</td>
                  
                  <td>0.333</td>
                  
                  <td>0.112</td>
                  
                  <td>0.05</td>
                  
                  <td>0.053</td>
                  
                  <td>0.063</td>
                  
                  <td>0.093</td>
                  
                  <td>0.073</td>
                  
                  <td>0.256</td>
                  
                  <td>0.072</td>
                  
                  <td>[0.0,0.0,0.667,0.333,0.112,0.05,0.053,0.063,0.093,0.073,0.256,0.072]</td>
                  
                  <td>[0.17390505926904495,0.009368407266899609,0.5191378491682284]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>291</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>29</td>
                  
                  <td>9</td>
                  
                  <td>56</td>
                  
                  <td>15</td>
                  
                  <td>34</td>
                  
                  <td>1092</td>
                  
                  <td>137</td>
                  
                  <td>283220</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.155</td>
                  
                  <td>0.113</td>
                  
                  <td>0.111</td>
                  
                  <td>0.095</td>
                  
                  <td>0.137</td>
                  
                  <td>0.133</td>
                  
                  <td>0.351</td>
                  
                  <td>0.141</td>
                  
                  <td>[1.0,0.0,0.333,0.333,0.155,0.113,0.111,0.095,0.137,0.133,0.351,0.141]</td>
                  
                  <td>[0.2137210316876795,1.0171183644215458,0.4654663950840127]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>44</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>19</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>3</td>
                  
                  <td>3</td>
                  
                  <td>196</td>
                  
                  <td>43</td>
                  
                  <td>48673</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.102</td>
                  
                  <td>0.013</td>
                  
                  <td>0.016</td>
                  
                  <td>0.019</td>
                  
                  <td>0.012</td>
                  
                  <td>0.024</td>
                  
                  <td>0.11</td>
                  
                  <td>0.024</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.102,0.013,0.016,0.019,0.012,0.024,0.11,0.024]</td>
                  
                  <td>[0.01805014396916193,0.003475538611471221,0.05600865937511797]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300051</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>4</td>
                  
                  <td>12</td>
                  
                  <td>85</td>
                  
                  <td>26</td>
                  
                  <td>33</td>
                  
                  <td>869</td>
                  
                  <td>26</td>
                  
                  <td>210807</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.021</td>
                  
                  <td>0.15</td>
                  
                  <td>0.168</td>
                  
                  <td>0.165</td>
                  
                  <td>0.133</td>
                  
                  <td>0.106</td>
                  
                  <td>0.067</td>
                  
                  <td>0.105</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.021,0.15,0.168,0.165,0.133,0.106,0.067,0.105]</td>
                  
                  <td>[1.0960754018982743,0.9966784684756574,-0.10461669543898444]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>154</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>9</td>
                  
                  <td>2</td>
                  
                  <td>4</td>
                  
                  <td>4</td>
                  
                  <td>4</td>
                  
                  <td>93</td>
                  
                  <td>11</td>
                  
                  <td>22515</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.048</td>
                  
                  <td>0.025</td>
                  
                  <td>0.008</td>
                  
                  <td>0.025</td>
                  
                  <td>0.016</td>
                  
                  <td>0.011</td>
                  
                  <td>0.028</td>
                  
                  <td>0.011</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.048,0.025,0.008,0.025,0.016,0.011,0.028,0.011]</td>
                  
                  <td>[0.01646385162922434,1.0010013211681366,0.00037198134038652665]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>22</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>62</td>
                  
                  <td>37</td>
                  
                  <td>14242</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.027</td>
                  
                  <td>0.0</td>
                  
                  <td>0.002</td>
                  
                  <td>0.006</td>
                  
                  <td>0.004</td>
                  
                  <td>0.007</td>
                  
                  <td>0.095</td>
                  
                  <td>0.007</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.027,0.0,0.002,0.006,0.004,0.007,0.095,0.007]</td>
                  
                  <td>[0.00465763881974591,0.0016981764111957392,0.016733169105237852]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>250</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>30</td>
                  
                  <td>40</td>
                  
                  <td>219</td>
                  
                  <td>74</td>
                  
                  <td>139</td>
                  
                  <td>4773</td>
                  
                  <td>77</td>
                  
                  <td>1185276</td>
                  
                  <td>0.667</td>
                  
                  <td>0.667</td>
                  
                  <td>0.16</td>
                  
                  <td>0.5</td>
                  
                  <td>0.433</td>
                  
                  <td>0.468</td>
                  
                  <td>0.56</td>
                  
                  <td>0.584</td>
                  
                  <td>0.197</td>
                  
                  <td>0.592</td>
                  
                  <td>[1.0,0.0,0.667,0.667,0.16,0.5,0.433,0.468,0.56,0.584,0.197,0.592]</td>
                  
                  <td>[0.706462905035945,1.0513677620123951,1.3860994428737823]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>116</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>12</td>
                  
                  <td>0</td>
                  
                  <td>6</td>
                  
                  <td>5</td>
                  
                  <td>1</td>
                  
                  <td>106</td>
                  
                  <td>101</td>
                  
                  <td>46967</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.064</td>
                  
                  <td>0.0</td>
                  
                  <td>0.012</td>
                  
                  <td>0.032</td>
                  
                  <td>0.004</td>
                  
                  <td>0.013</td>
                  
                  <td>0.259</td>
                  
                  <td>0.023</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.064,0.0,0.012,0.032,0.004,0.013,0.259,0.023]</td>
                  
                  <td>[0.014874539666925791,0.00470254816597599,0.047068195630280744]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200012</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>41</td>
                  
                  <td>63</td>
                  
                  <td>11053</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.013</td>
                  
                  <td>0.002</td>
                  
                  <td>0.0</td>
                  
                  <td>0.012</td>
                  
                  <td>0.005</td>
                  
                  <td>0.162</td>
                  
                  <td>0.005</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.0,0.013,0.002,0.0,0.012,0.005,0.162,0.005]</td>
                  
                  <td>[0.00755370546634554,1.001531610481225,-0.02062156037323492]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>276</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>6</td>
                  
                  <td>43</td>
                  
                  <td>220</td>
                  
                  <td>61</td>
                  
                  <td>140</td>
                  
                  <td>4516</td>
                  
                  <td>76</td>
                  
                  <td>1116916</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.032</td>
                  
                  <td>0.537</td>
                  
                  <td>0.435</td>
                  
                  <td>0.386</td>
                  
                  <td>0.565</td>
                  
                  <td>0.552</td>
                  
                  <td>0.195</td>
                  
                  <td>0.558</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.032,0.537,0.435,0.386,0.565,0.552,0.195,0.558]</td>
                  
                  <td>[1.4670580982297803,1.029367852729706,0.5225810185647611]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>118</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>38</td>
                  
                  <td>12</td>
                  
                  <td>103</td>
                  
                  <td>36</td>
                  
                  <td>62</td>
                  
                  <td>2170</td>
                  
                  <td>60</td>
                  
                  <td>536384</td>
                  
                  <td>0.333</td>
                  
                  <td>0.667</td>
                  
                  <td>0.203</td>
                  
                  <td>0.15</td>
                  
                  <td>0.204</td>
                  
                  <td>0.228</td>
                  
                  <td>0.25</td>
                  
                  <td>0.265</td>
                  
                  <td>0.154</td>
                  
                  <td>0.268</td>
                  
                  <td>[0.0,1.0,0.333,0.667,0.203,0.15,0.204,0.228,0.25,0.265,0.154,0.268]</td>
                  
                  <td>[1.275502458814086,0.009574301335914696,0.37442468117213146]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>187</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>10</td>
                  
                  <td>3</td>
                  
                  <td>10</td>
                  
                  <td>7</td>
                  
                  <td>3</td>
                  
                  <td>142</td>
                  
                  <td>21</td>
                  
                  <td>33942</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.053</td>
                  
                  <td>0.037</td>
                  
                  <td>0.02</td>
                  
                  <td>0.044</td>
                  
                  <td>0.012</td>
                  
                  <td>0.017</td>
                  
                  <td>0.054</td>
                  
                  <td>0.017</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.053,0.037,0.02,0.044,0.012,0.017,0.054,0.017]</td>
                  
                  <td>[0.024618210711044232,0.0032181106519115535,0.05537986021032179]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>190</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>3</td>
                  
                  <td>8</td>
                  
                  <td>29</td>
                  
                  <td>14</td>
                  
                  <td>19</td>
                  
                  <td>783</td>
                  
                  <td>67</td>
                  
                  <td>201372</td>
                  
                  <td>0.333</td>
                  
                  <td>0.667</td>
                  
                  <td>0.016</td>
                  
                  <td>0.1</td>
                  
                  <td>0.057</td>
                  
                  <td>0.089</td>
                  
                  <td>0.077</td>
                  
                  <td>0.096</td>
                  
                  <td>0.172</td>
                  
                  <td>0.1</td>
                  
                  <td>[1.0,1.0,0.333,0.667,0.016,0.1,0.057,0.089,0.077,0.096,0.172,0.1]</td>
                  
                  <td>[1.1367750397777336,0.9969252973605156,0.06831380089179989]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>290</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>22</td>
                  
                  <td>6</td>
                  
                  <td>42</td>
                  
                  <td>24</td>
                  
                  <td>13</td>
                  
                  <td>601</td>
                  
                  <td>91</td>
                  
                  <td>150373</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.118</td>
                  
                  <td>0.075</td>
                  
                  <td>0.083</td>
                  
                  <td>0.152</td>
                  
                  <td>0.052</td>
                  
                  <td>0.073</td>
                  
                  <td>0.233</td>
                  
                  <td>0.075</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.118,0.075,0.083,0.152,0.052,0.073,0.233,0.075]</td>
                  
                  <td>[1.0436811596172944,0.994979447262617,-0.16611615872966304]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>129</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>54</td>
                  
                  <td>19</td>
                  
                  <td>99</td>
                  
                  <td>23</td>
                  
                  <td>59</td>
                  
                  <td>1941</td>
                  
                  <td>61</td>
                  
                  <td>476264</td>
                  
                  <td>0.667</td>
                  
                  <td>0.667</td>
                  
                  <td>0.289</td>
                  
                  <td>0.237</td>
                  
                  <td>0.196</td>
                  
                  <td>0.146</td>
                  
                  <td>0.238</td>
                  
                  <td>0.237</td>
                  
                  <td>0.156</td>
                  
                  <td>0.238</td>
                  
                  <td>[1.0,0.0,0.667,0.667,0.289,0.237,0.196,0.146,0.238,0.237,0.156,0.238]</td>
                  
                  <td>[0.3969514502207533,1.0264306058031576,0.8982295297651413]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>274</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>19</td>
                  
                  <td>3</td>
                  
                  <td>20</td>
                  
                  <td>12</td>
                  
                  <td>16</td>
                  
                  <td>501</td>
                  
                  <td>70</td>
                  
                  <td>127271</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.102</td>
                  
                  <td>0.037</td>
                  
                  <td>0.04</td>
                  
                  <td>0.076</td>
                  
                  <td>0.065</td>
                  
                  <td>0.061</td>
                  
                  <td>0.179</td>
                  
                  <td>0.063</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.102,0.037,0.04,0.076,0.065,0.061,0.179,0.063]</td>
                  
                  <td>[1.0150769971785851,-0.00702117509658681,-0.18077571913044305]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>37</td>
                  
                  <td>143</td>
                  
                  <td>71</td>
                  
                  <td>94</td>
                  
                  <td>3382</td>
                  
                  <td>65</td>
                  
                  <td>820118</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.043</td>
                  
                  <td>0.463</td>
                  
                  <td>0.283</td>
                  
                  <td>0.449</td>
                  
                  <td>0.379</td>
                  
                  <td>0.414</td>
                  
                  <td>0.167</td>
                  
                  <td>0.41</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.043,0.463,0.283,0.449,0.379,0.414,0.167,0.41]</td>
                  
                  <td>[1.3610561798964314,1.0210750137659284,0.34964679521726577]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>155</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>7</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>140</td>
                  
                  <td>95</td>
                  
                  <td>34578</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.014</td>
                  
                  <td>0.0</td>
                  
                  <td>0.016</td>
                  
                  <td>0.017</td>
                  
                  <td>0.244</td>
                  
                  <td>0.017</td>
                  
                  <td>(12,[1,6,8,9,10,11],[1.0,0.014,0.016,0.017,0.244,0.017])</td>
                  
                  <td>[0.8948887685627348,-0.016410235671716556,-0.40976358056776047]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>81</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>16</td>
                  
                  <td>81</td>
                  
                  <td>17</td>
                  
                  <td>48</td>
                  
                  <td>1643</td>
                  
                  <td>96</td>
                  
                  <td>403120</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.005</td>
                  
                  <td>0.2</td>
                  
                  <td>0.16</td>
                  
                  <td>0.108</td>
                  
                  <td>0.194</td>
                  
                  <td>0.201</td>
                  
                  <td>0.246</td>
                  
                  <td>0.201</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.005,0.2,0.16,0.108,0.194,0.201,0.246,0.201]</td>
                  
                  <td>[1.063179030148533,0.9967014527060993,-0.16503166349890283]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>266</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>1</td>
                  
                  <td>22</td>
                  
                  <td>85</td>
                  
                  <td>5309</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.011</td>
                  
                  <td>0.0</td>
                  
                  <td>0.004</td>
                  
                  <td>0.013</td>
                  
                  <td>0.004</td>
                  
                  <td>0.003</td>
                  
                  <td>0.218</td>
                  
                  <td>0.002</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.011,0.0,0.004,0.013,0.004,0.003,0.218,0.002]</td>
                  
                  <td>[0.005279551304214255,0.003153609060257881,0.018514614457962104]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>222</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>59</td>
                  
                  <td>4</td>
                  
                  <td>29</td>
                  
                  <td>10</td>
                  
                  <td>13</td>
                  
                  <td>560</td>
                  
                  <td>67</td>
                  
                  <td>139996</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.316</td>
                  
                  <td>0.05</td>
                  
                  <td>0.057</td>
                  
                  <td>0.063</td>
                  
                  <td>0.052</td>
                  
                  <td>0.068</td>
                  
                  <td>0.172</td>
                  
                  <td>0.07</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.316,0.05,0.057,0.063,0.052,0.068,0.172,0.07]</td>
                  
                  <td>[0.059206872333749036,0.00902971519065186,0.17350423121591532]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>12</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>18</td>
                  
                  <td>39</td>
                  
                  <td>172</td>
                  
                  <td>60</td>
                  
                  <td>104</td>
                  
                  <td>3548</td>
                  
                  <td>112</td>
                  
                  <td>980137</td>
                  
                  <td>0.333</td>
                  
                  <td>0.667</td>
                  
                  <td>0.096</td>
                  
                  <td>0.487</td>
                  
                  <td>0.34</td>
                  
                  <td>0.38</td>
                  
                  <td>0.419</td>
                  
                  <td>0.434</td>
                  
                  <td>0.287</td>
                  
                  <td>0.49</td>
                  
                  <td>[0.0,1.0,0.333,0.667,0.096,0.487,0.34,0.38,0.419,0.434,0.287,0.49]</td>
                  
                  <td>[1.477053875794933,0.028742204082318198,0.6999737462491381]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300002</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>16</td>
                  
                  <td>98</td>
                  
                  <td>22</td>
                  
                  <td>23</td>
                  
                  <td>1103</td>
                  
                  <td>126</td>
                  
                  <td>272278</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.011</td>
                  
                  <td>0.2</td>
                  
                  <td>0.194</td>
                  
                  <td>0.139</td>
                  
                  <td>0.093</td>
                  
                  <td>0.135</td>
                  
                  <td>0.323</td>
                  
                  <td>0.136</td>
                  
                  <td>[0.0,1.0,0.333,0.333,0.011,0.2,0.194,0.139,0.093,0.135,0.323,0.136]</td>
                  
                  <td>[1.1250651895844948,-0.0006291436220726349,0.08527066818194057]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>206</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>14</td>
                  
                  <td>10</td>
                  
                  <td>55</td>
                  
                  <td>24</td>
                  
                  <td>30</td>
                  
                  <td>1213</td>
                  
                  <td>143</td>
                  
                  <td>307694</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.075</td>
                  
                  <td>0.125</td>
                  
                  <td>0.109</td>
                  
                  <td>0.152</td>
                  
                  <td>0.121</td>
                  
                  <td>0.148</td>
                  
                  <td>0.367</td>
                  
                  <td>0.153</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.075,0.125,0.109,0.152,0.121,0.148,0.367,0.153]</td>
                  
                  <td>[1.0948899534937304,1.000604323479496,-0.08551849154987098]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>82</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>3</td>
                  
                  <td>7</td>
                  
                  <td>70</td>
                  
                  <td>42</td>
                  
                  <td>37</td>
                  
                  <td>1171</td>
                  
                  <td>77</td>
                  
                  <td>286792</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.016</td>
                  
                  <td>0.087</td>
                  
                  <td>0.138</td>
                  
                  <td>0.266</td>
                  
                  <td>0.149</td>
                  
                  <td>0.143</td>
                  
                  <td>0.197</td>
                  
                  <td>0.143</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.016,0.087,0.138,0.266,0.149,0.143,0.197,0.143]</td>
                  
                  <td>[1.1123043579675913,0.00020373424052449194,-0.03888385083455342]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100036</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>14</td>
                  
                  <td>1</td>
                  
                  <td>6</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>92</td>
                  
                  <td>88</td>
                  
                  <td>22590</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.075</td>
                  
                  <td>0.013</td>
                  
                  <td>0.012</td>
                  
                  <td>0.006</td>
                  
                  <td>0.0</td>
                  
                  <td>0.011</td>
                  
                  <td>0.226</td>
                  
                  <td>0.011</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.075,0.013,0.012,0.006,0.0,0.011,0.226,0.011]</td>
                  
                  <td>[0.010122423645458909,1.0029580579526411,0.002139785539803855]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>193</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>1</td>
                  
                  <td>4</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>113</td>
                  
                  <td>120</td>
                  
                  <td>25433</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.021</td>
                  
                  <td>0.013</td>
                  
                  <td>0.008</td>
                  
                  <td>0.0</td>
                  
                  <td>0.02</td>
                  
                  <td>0.014</td>
                  
                  <td>0.308</td>
                  
                  <td>0.012</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.021,0.013,0.008,0.0,0.02,0.014,0.308,0.012]</td>
                  
                  <td>[0.012655859105390195,0.004999609827885892,0.03638932099508968]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>157</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>32</td>
                  
                  <td>27</td>
                  
                  <td>115</td>
                  
                  <td>44</td>
                  
                  <td>54</td>
                  
                  <td>2500</td>
                  
                  <td>105</td>
                  
                  <td>614750</td>
                  
                  <td>0.333</td>
                  
                  <td>0.667</td>
                  
                  <td>0.171</td>
                  
                  <td>0.338</td>
                  
                  <td>0.227</td>
                  
                  <td>0.278</td>
                  
                  <td>0.218</td>
                  
                  <td>0.306</td>
                  
                  <td>0.269</td>
                  
                  <td>0.307</td>
                  
                  <td>[0.0,1.0,0.333,0.667,0.171,0.338,0.227,0.278,0.218,0.306,0.269,0.307]</td>
                  
                  <td>[1.3298886667573042,0.016740073499085448,0.4698977009627345]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>120</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>16</td>
                  
                  <td>83</td>
                  
                  <td>15</td>
                  
                  <td>43</td>
                  
                  <td>1262</td>
                  
                  <td>116</td>
                  
                  <td>313138</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.011</td>
                  
                  <td>0.2</td>
                  
                  <td>0.164</td>
                  
                  <td>0.095</td>
                  
                  <td>0.173</td>
                  
                  <td>0.154</td>
                  
                  <td>0.297</td>
                  
                  <td>0.156</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.011,0.2,0.164,0.095,0.173,0.154,0.297,0.156]</td>
                  
                  <td>[1.0422869662459828,-0.0030453512448098247,-0.15659837198949722]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>11</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>10</td>
                  
                  <td>1</td>
                  
                  <td>13</td>
                  
                  <td>6</td>
                  
                  <td>5</td>
                  
                  <td>194</td>
                  
                  <td>122</td>
                  
                  <td>47752</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.053</td>
                  
                  <td>0.013</td>
                  
                  <td>0.026</td>
                  
                  <td>0.038</td>
                  
                  <td>0.02</td>
                  
                  <td>0.024</td>
                  
                  <td>0.313</td>
                  
                  <td>0.023</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.053,0.013,0.026,0.038,0.02,0.024,0.313,0.023]</td>
                  
                  <td>[0.9829143371410412,-0.008366566349417504,-0.2402121705969416]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>234</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>30</td>
                  
                  <td>22</td>
                  
                  <td>91</td>
                  
                  <td>17</td>
                  
                  <td>51</td>
                  
                  <td>1674</td>
                  
                  <td>36</td>
                  
                  <td>412581</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.16</td>
                  
                  <td>0.275</td>
                  
                  <td>0.18</td>
                  
                  <td>0.108</td>
                  
                  <td>0.206</td>
                  
                  <td>0.205</td>
                  
                  <td>0.092</td>
                  
                  <td>0.206</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.16,0.275,0.18,0.108,0.206,0.205,0.092,0.206]</td>
                  
                  <td>[1.1561465748461102,1.0036565396672554,0.03039948041468618]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>166</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>59</td>
                  
                  <td>8</td>
                  
                  <td>34</td>
                  
                  <td>27</td>
                  
                  <td>30</td>
                  
                  <td>683</td>
                  
                  <td>101</td>
                  
                  <td>171718</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.316</td>
                  
                  <td>0.1</td>
                  
                  <td>0.067</td>
                  
                  <td>0.171</td>
                  
                  <td>0.121</td>
                  
                  <td>0.083</td>
                  
                  <td>0.259</td>
                  
                  <td>0.085</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.316,0.1,0.067,0.171,0.121,0.083,0.259,0.085]</td>
                  
                  <td>[0.10445148075407369,1.0130042303440392,0.2135871533413423]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300012</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>2</td>
                  
                  <td>29</td>
                  
                  <td>2</td>
                  
                  <td>8</td>
                  
                  <td>252</td>
                  
                  <td>152</td>
                  
                  <td>61666</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.043</td>
                  
                  <td>0.025</td>
                  
                  <td>0.057</td>
                  
                  <td>0.013</td>
                  
                  <td>0.032</td>
                  
                  <td>0.031</td>
                  
                  <td>0.39</td>
                  
                  <td>0.03</td>
                  
                  <td>[1.0,0.0,0.0,0.333,0.043,0.025,0.057,0.013,0.032,0.031,0.39,0.03]</td>
                  
                  <td>[0.10804836967081784,1.0121522808851964,0.17204115805394984]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100033</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>4</td>
                  
                  <td>22</td>
                  
                  <td>12</td>
                  
                  <td>21</td>
                  
                  <td>700</td>
                  
                  <td>95</td>
                  
                  <td>181397</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.021</td>
                  
                  <td>0.05</td>
                  
                  <td>0.043</td>
                  
                  <td>0.076</td>
                  
                  <td>0.085</td>
                  
                  <td>0.085</td>
                  
                  <td>0.244</td>
                  
                  <td>0.09</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.021,0.05,0.043,0.076,0.085,0.085,0.244,0.09]</td>
                  
                  <td>[0.9558942087765995,-0.011192898206311561,-0.3037035255290661]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200042</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>13</td>
                  
                  <td>5</td>
                  
                  <td>6</td>
                  
                  <td>5</td>
                  
                  <td>5</td>
                  
                  <td>154</td>
                  
                  <td>82</td>
                  
                  <td>35712</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.07</td>
                  
                  <td>0.062</td>
                  
                  <td>0.012</td>
                  
                  <td>0.032</td>
                  
                  <td>0.02</td>
                  
                  <td>0.019</td>
                  
                  <td>0.21</td>
                  
                  <td>0.017</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.07,0.062,0.012,0.032,0.02,0.019,0.21,0.017]</td>
                  
                  <td>[0.028611609111461055,1.0046885291677161,0.033193906926156105]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300010</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>10</td>
                  
                  <td>9</td>
                  
                  <td>88</td>
                  
                  <td>20</td>
                  
                  <td>43</td>
                  
                  <td>1022</td>
                  
                  <td>78</td>
                  
                  <td>254275</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.053</td>
                  
                  <td>0.113</td>
                  
                  <td>0.174</td>
                  
                  <td>0.127</td>
                  
                  <td>0.173</td>
                  
                  <td>0.125</td>
                  
                  <td>0.2</td>
                  
                  <td>0.127</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.053,0.113,0.174,0.127,0.173,0.125,0.2,0.127]</td>
                  
                  <td>[1.0982204190241942,0.9982558086138278,-0.09180782670846657]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>279</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>11</td>
                  
                  <td>3</td>
                  
                  <td>22</td>
                  
                  <td>8</td>
                  
                  <td>14</td>
                  
                  <td>496</td>
                  
                  <td>226</td>
                  
                  <td>120445</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.059</td>
                  
                  <td>0.037</td>
                  
                  <td>0.043</td>
                  
                  <td>0.051</td>
                  
                  <td>0.056</td>
                  
                  <td>0.061</td>
                  
                  <td>0.579</td>
                  
                  <td>0.06</td>
                  
                  <td>[0.0,1.0,0.333,0.333,0.059,0.037,0.043,0.051,0.056,0.061,0.579,0.06]</td>
                  
                  <td>[1.0273305197287375,-0.005558951471707016,-0.06201538381540635]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>204</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>22</td>
                  
                  <td>12</td>
                  
                  <td>6</td>
                  
                  <td>339</td>
                  
                  <td>124</td>
                  
                  <td>84632</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.062</td>
                  
                  <td>0.043</td>
                  
                  <td>0.076</td>
                  
                  <td>0.024</td>
                  
                  <td>0.041</td>
                  
                  <td>0.318</td>
                  
                  <td>0.042</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.0,0.062,0.043,0.076,0.024,0.041,0.318,0.042]</td>
                  
                  <td>[0.9329120071997755,-0.012052232881665698,-0.34215210973397486]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>240</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>7</td>
                  
                  <td>0</td>
                  
                  <td>6</td>
                  
                  <td>3</td>
                  
                  <td>3</td>
                  
                  <td>99</td>
                  
                  <td>58</td>
                  
                  <td>23310</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.037</td>
                  
                  <td>0.0</td>
                  
                  <td>0.012</td>
                  
                  <td>0.019</td>
                  
                  <td>0.012</td>
                  
                  <td>0.012</td>
                  
                  <td>0.149</td>
                  
                  <td>0.011</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.037,0.0,0.012,0.019,0.012,0.012,0.149,0.011]</td>
                  
                  <td>[0.01140126187137193,0.002907955275791374,0.03168520188605277]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300016</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>23</td>
                  
                  <td>7</td>
                  
                  <td>105</td>
                  
                  <td>19</td>
                  
                  <td>38</td>
                  
                  <td>1228</td>
                  
                  <td>104</td>
                  
                  <td>302066</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.123</td>
                  
                  <td>0.087</td>
                  
                  <td>0.208</td>
                  
                  <td>0.12</td>
                  
                  <td>0.153</td>
                  
                  <td>0.15</td>
                  
                  <td>0.267</td>
                  
                  <td>0.151</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.123,0.087,0.208,0.12,0.153,0.15,0.267,0.151]</td>
                  
                  <td>[1.102942770449742,0.9994074213625859,-0.0681714141453853]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>271</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>8</td>
                  
                  <td>2</td>
                  
                  <td>3</td>
                  
                  <td>190</td>
                  
                  <td>36</td>
                  
                  <td>47401</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.037</td>
                  
                  <td>0.016</td>
                  
                  <td>0.013</td>
                  
                  <td>0.012</td>
                  
                  <td>0.023</td>
                  
                  <td>0.092</td>
                  
                  <td>0.023</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.0,0.037,0.016,0.013,0.012,0.023,0.092,0.023]</td>
                  
                  <td>[0.9050110276045167,0.9816975513775985,-0.43570955294633706]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>195</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>40</td>
                  
                  <td>53</td>
                  
                  <td>253</td>
                  
                  <td>90</td>
                  
                  <td>147</td>
                  
                  <td>5158</td>
                  
                  <td>86</td>
                  
                  <td>1261818</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.214</td>
                  
                  <td>0.662</td>
                  
                  <td>0.5</td>
                  
                  <td>0.57</td>
                  
                  <td>0.593</td>
                  
                  <td>0.631</td>
                  
                  <td>0.221</td>
                  
                  <td>0.63</td>
                  
                  <td>[1.0,1.0,0.333,0.333,0.214,0.662,0.5,0.57,0.593,0.631,0.221,0.63]</td>
                  
                  <td>[1.574542284961798,1.0361374027776968,0.8389471466015996]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100005</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>17</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>72</td>
                  
                  <td>72</td>
                  
                  <td>16684</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.091</td>
                  
                  <td>0.013</td>
                  
                  <td>0.004</td>
                  
                  <td>0.0</td>
                  
                  <td>0.004</td>
                  
                  <td>0.009</td>
                  
                  <td>0.185</td>
                  
                  <td>0.008</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.091,0.013,0.004,0.0,0.004,0.009,0.185,0.008]</td>
                  
                  <td>[0.00736902196286869,1.0023955460392462,-0.00021542196769070918]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300005</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>29</td>
                  
                  <td>9</td>
                  
                  <td>17</td>
                  
                  <td>407</td>
                  
                  <td>156</td>
                  
                  <td>100423</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.025</td>
                  
                  <td>0.057</td>
                  
                  <td>0.057</td>
                  
                  <td>0.069</td>
                  
                  <td>0.05</td>
                  
                  <td>0.4</td>
                  
                  <td>0.05</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.0,0.025,0.057,0.057,0.069,0.05,0.4,0.05]</td>
                  
                  <td>[0.9360406489965425,-0.011247080391514596,-0.3364884149628848]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>170</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>30</td>
                  
                  <td>1</td>
                  
                  <td>23</td>
                  
                  <td>3</td>
                  
                  <td>15</td>
                  
                  <td>411</td>
                  
                  <td>67</td>
                  
                  <td>100574</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.16</td>
                  
                  <td>0.013</td>
                  
                  <td>0.045</td>
                  
                  <td>0.019</td>
                  
                  <td>0.06</td>
                  
                  <td>0.05</td>
                  
                  <td>0.172</td>
                  
                  <td>0.05</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.16,0.013,0.045,0.019,0.06,0.05,0.172,0.05]</td>
                  
                  <td>[0.03972891442001499,1.0049803406862357,0.06656500773366344]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>71</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>5</td>
                  
                  <td>0</td>
                  
                  <td>69</td>
                  
                  <td>62</td>
                  
                  <td>15856</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.027</td>
                  
                  <td>0.013</td>
                  
                  <td>0.004</td>
                  
                  <td>0.032</td>
                  
                  <td>0.0</td>
                  
                  <td>0.008</td>
                  
                  <td>0.159</td>
                  
                  <td>0.008</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.027,0.013,0.004,0.032,0.0,0.008,0.159,0.008]</td>
                  
                  <td>[0.012019281657143043,1.0020206322262049,-0.007487863752322799]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>133</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>44</td>
                  
                  <td>77</td>
                  
                  <td>12971</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.016</td>
                  
                  <td>0.013</td>
                  
                  <td>0.01</td>
                  
                  <td>0.0</td>
                  
                  <td>0.004</td>
                  
                  <td>0.005</td>
                  
                  <td>0.197</td>
                  
                  <td>0.006</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.016,0.013,0.01,0.0,0.004,0.005,0.197,0.006]</td>
                  
                  <td>[0.007783318121928738,1.0020381315370985,-0.01556847944175206]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>94</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>3</td>
                  
                  <td>4</td>
                  
                  <td>4</td>
                  
                  <td>3</td>
                  
                  <td>101</td>
                  
                  <td>117</td>
                  
                  <td>23753</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.027</td>
                  
                  <td>0.037</td>
                  
                  <td>0.008</td>
                  
                  <td>0.025</td>
                  
                  <td>0.012</td>
                  
                  <td>0.012</td>
                  
                  <td>0.3</td>
                  
                  <td>0.011</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.027,0.037,0.008,0.025,0.012,0.012,0.3,0.011]</td>
                  
                  <td>[0.9768825366262929,-0.008821712762150767,-0.2547126443662796]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>110</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>39</td>
                  
                  <td>2</td>
                  
                  <td>19</td>
                  
                  <td>5</td>
                  
                  <td>9</td>
                  
                  <td>417</td>
                  
                  <td>68</td>
                  
                  <td>136443</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.209</td>
                  
                  <td>0.025</td>
                  
                  <td>0.038</td>
                  
                  <td>0.032</td>
                  
                  <td>0.036</td>
                  
                  <td>0.051</td>
                  
                  <td>0.174</td>
                  
                  <td>0.068</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.209,0.025,0.038,0.032,0.036,0.051,0.174,0.068]</td>
                  
                  <td>[0.042015018827276734,1.005644342572785,0.0818161928849934]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>132</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>1</td>
                  
                  <td>4</td>
                  
                  <td>4</td>
                  
                  <td>4</td>
                  
                  <td>113</td>
                  
                  <td>23</td>
                  
                  <td>26142</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.027</td>
                  
                  <td>0.013</td>
                  
                  <td>0.008</td>
                  
                  <td>0.025</td>
                  
                  <td>0.016</td>
                  
                  <td>0.014</td>
                  
                  <td>0.059</td>
                  
                  <td>0.013</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.027,0.013,0.008,0.025,0.016,0.014,0.059,0.013]</td>
                  
                  <td>[0.9727595357253175,-0.012354569268144969,-0.27106872027245005]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100002</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>2</td>
                  
                  <td>8</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>137</td>
                  
                  <td>130</td>
                  
                  <td>36063</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.027</td>
                  
                  <td>0.025</td>
                  
                  <td>0.016</td>
                  
                  <td>0.0</td>
                  
                  <td>0.012</td>
                  
                  <td>0.017</td>
                  
                  <td>0.333</td>
                  
                  <td>0.018</td>
                  
                  <td>[0.0,0.0,0.333,0.333,0.027,0.025,0.016,0.0,0.012,0.017,0.333,0.018]</td>
                  
                  <td>[0.10669233041788052,0.008545236061619265,0.29829290817138604]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>79</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>17</td>
                  
                  <td>1</td>
                  
                  <td>9</td>
                  
                  <td>6</td>
                  
                  <td>8</td>
                  
                  <td>213</td>
                  
                  <td>68</td>
                  
                  <td>52760</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.091</td>
                  
                  <td>0.013</td>
                  
                  <td>0.018</td>
                  
                  <td>0.038</td>
                  
                  <td>0.032</td>
                  
                  <td>0.026</td>
                  
                  <td>0.174</td>
                  
                  <td>0.026</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.091,0.013,0.018,0.038,0.032,0.026,0.174,0.026]</td>
                  
                  <td>[0.026240623429825066,1.0036670431820431,0.0299530041461801]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>6</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>46</td>
                  
                  <td>15</td>
                  
                  <td>55</td>
                  
                  <td>9</td>
                  
                  <td>35</td>
                  
                  <td>1127</td>
                  
                  <td>257</td>
                  
                  <td>278332</td>
                  
                  <td>0.333</td>
                  
                  <td>0.667</td>
                  
                  <td>0.246</td>
                  
                  <td>0.188</td>
                  
                  <td>0.109</td>
                  
                  <td>0.057</td>
                  
                  <td>0.141</td>
                  
                  <td>0.138</td>
                  
                  <td>0.659</td>
                  
                  <td>0.139</td>
                  
                  <td>[1.0,1.0,0.333,0.667,0.246,0.188,0.109,0.057,0.141,0.138,0.659,0.139]</td>
                  
                  <td>[1.1806344039597503,1.0084707190655147,0.21091466123477587]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>34</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>10</td>
                  
                  <td>11</td>
                  
                  <td>59</td>
                  
                  <td>19</td>
                  
                  <td>32</td>
                  
                  <td>1239</td>
                  
                  <td>77</td>
                  
                  <td>306096</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.053</td>
                  
                  <td>0.138</td>
                  
                  <td>0.117</td>
                  
                  <td>0.12</td>
                  
                  <td>0.129</td>
                  
                  <td>0.151</td>
                  
                  <td>0.197</td>
                  
                  <td>0.153</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.053,0.138,0.117,0.12,0.129,0.151,0.197,0.153]</td>
                  
                  <td>[1.0943044560477335,0.9983909336661029,-0.0965967921316733]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>127</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>8</td>
                  
                  <td>8</td>
                  
                  <td>98</td>
                  
                  <td>31</td>
                  
                  <td>48</td>
                  
                  <td>1921</td>
                  
                  <td>74</td>
                  
                  <td>470494</td>
                  
                  <td>0.667</td>
                  
                  <td>0.667</td>
                  
                  <td>0.043</td>
                  
                  <td>0.1</td>
                  
                  <td>0.194</td>
                  
                  <td>0.196</td>
                  
                  <td>0.194</td>
                  
                  <td>0.235</td>
                  
                  <td>0.19</td>
                  
                  <td>0.235</td>
                  
                  <td>[0.0,1.0,0.667,0.667,0.043,0.1,0.194,0.196,0.194,0.235,0.19,0.235]</td>
                  
                  <td>[1.2567565601708477,0.002909341268865622,0.40243501184018327]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>80</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>12</td>
                  
                  <td>10</td>
                  
                  <td>58</td>
                  
                  <td>26</td>
                  
                  <td>29</td>
                  
                  <td>1121</td>
                  
                  <td>55</td>
                  
                  <td>275551</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.064</td>
                  
                  <td>0.125</td>
                  
                  <td>0.115</td>
                  
                  <td>0.165</td>
                  
                  <td>0.117</td>
                  
                  <td>0.137</td>
                  
                  <td>0.141</td>
                  
                  <td>0.137</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.064,0.125,0.115,0.165,0.117,0.137,0.141,0.137]</td>
                  
                  <td>[1.0911369033156322,-0.0013719415112445846,-0.06264162750478128]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200025</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>33</td>
                  
                  <td>27</td>
                  
                  <td>38</td>
                  
                  <td>18</td>
                  
                  <td>32</td>
                  
                  <td>984</td>
                  
                  <td>122</td>
                  
                  <td>238128</td>
                  
                  <td>0.667</td>
                  
                  <td>0.667</td>
                  
                  <td>0.176</td>
                  
                  <td>0.338</td>
                  
                  <td>0.075</td>
                  
                  <td>0.114</td>
                  
                  <td>0.129</td>
                  
                  <td>0.12</td>
                  
                  <td>0.313</td>
                  
                  <td>0.119</td>
                  
                  <td>[1.0,0.0,0.667,0.667,0.176,0.338,0.075,0.114,0.129,0.12,0.313,0.119]</td>
                  
                  <td>[0.33393947175057115,1.024493648810816,0.7833023668807333]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>169</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>13</td>
                  
                  <td>0</td>
                  
                  <td>6</td>
                  
                  <td>6</td>
                  
                  <td>0</td>
                  
                  <td>112</td>
                  
                  <td>79</td>
                  
                  <td>26208</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.07</td>
                  
                  <td>0.0</td>
                  
                  <td>0.012</td>
                  
                  <td>0.038</td>
                  
                  <td>0.0</td>
                  
                  <td>0.014</td>
                  
                  <td>0.203</td>
                  
                  <td>0.013</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.07,0.0,0.012,0.038,0.0,0.014,0.203,0.013]</td>
                  
                  <td>[0.013815619514828785,1.0028115442439993,0.005604261773751393]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>39</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>26</td>
                  
                  <td>9</td>
                  
                  <td>25</td>
                  
                  <td>16</td>
                  
                  <td>20</td>
                  
                  <td>629</td>
                  
                  <td>115</td>
                  
                  <td>152020</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.139</td>
                  
                  <td>0.113</td>
                  
                  <td>0.049</td>
                  
                  <td>0.101</td>
                  
                  <td>0.081</td>
                  
                  <td>0.077</td>
                  
                  <td>0.295</td>
                  
                  <td>0.076</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.139,0.113,0.049,0.101,0.081,0.077,0.295,0.076]</td>
                  
                  <td>[1.0419452400844607,-0.0025182969930225506,-0.12095180747489576]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100032</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>7</td>
                  
                  <td>76</td>
                  
                  <td>1794</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.016</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.001</td>
                  
                  <td>0.195</td>
                  
                  <td>0.001</td>
                  
                  <td>(12,[0,4,9,10,11],[1.0,0.016,0.001,0.195,0.001])</td>
                  
                  <td>[0.0017045289821811516,1.0014667557759607,-0.026140270233623624]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>4</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>36</td>
                  
                  <td>6</td>
                  
                  <td>27</td>
                  
                  <td>7</td>
                  
                  <td>10</td>
                  
                  <td>421</td>
                  
                  <td>62</td>
                  
                  <td>97204</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.193</td>
                  
                  <td>0.075</td>
                  
                  <td>0.053</td>
                  
                  <td>0.044</td>
                  
                  <td>0.04</td>
                  
                  <td>0.051</td>
                  
                  <td>0.159</td>
                  
                  <td>0.048</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.193,0.075,0.053,0.044,0.04,0.051,0.159,0.048]</td>
                  
                  <td>[0.05239838492483136,1.0066250837996586,0.09734720855431458]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>24</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>29</td>
                  
                  <td>13</td>
                  
                  <td>69</td>
                  
                  <td>31</td>
                  
                  <td>29</td>
                  
                  <td>1297</td>
                  
                  <td>75</td>
                  
                  <td>321148</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.155</td>
                  
                  <td>0.163</td>
                  
                  <td>0.136</td>
                  
                  <td>0.196</td>
                  
                  <td>0.117</td>
                  
                  <td>0.159</td>
                  
                  <td>0.192</td>
                  
                  <td>0.16</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.155,0.163,0.136,0.196,0.117,0.159,0.192,0.16]</td>
                  
                  <td>[1.1141655522143796,1.0007481171845125,-0.04075891996201643]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200007</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>2</td>
                  
                  <td>10</td>
                  
                  <td>2</td>
                  
                  <td>6</td>
                  
                  <td>140</td>
                  
                  <td>58</td>
                  
                  <td>38821</td>
                  
                  <td>0.333</td>
                  
                  <td>0.0</td>
                  
                  <td>0.021</td>
                  
                  <td>0.025</td>
                  
                  <td>0.02</td>
                  
                  <td>0.013</td>
                  
                  <td>0.024</td>
                  
                  <td>0.017</td>
                  
                  <td>0.149</td>
                  
                  <td>0.019</td>
                  
                  <td>[0.0,1.0,0.333,0.0,0.021,0.025,0.02,0.013,0.024,0.017,0.149,0.019]</td>
                  
                  <td>[0.9190135070893033,-0.019407546852345095,-0.27344084329986146]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>168</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>34</td>
                  
                  <td>11</td>
                  
                  <td>55</td>
                  
                  <td>19</td>
                  
                  <td>43</td>
                  
                  <td>1266</td>
                  
                  <td>60</td>
                  
                  <td>308380</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.182</td>
                  
                  <td>0.138</td>
                  
                  <td>0.109</td>
                  
                  <td>0.12</td>
                  
                  <td>0.173</td>
                  
                  <td>0.155</td>
                  
                  <td>0.154</td>
                  
                  <td>0.154</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.182,0.138,0.109,0.12,0.173,0.155,0.154,0.154]</td>
                  
                  <td>[1.0995470663605842,0.0003933588682362764,-0.022224102948692787]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>59</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>40</td>
                  
                  <td>2</td>
                  
                  <td>17</td>
                  
                  <td>14</td>
                  
                  <td>15</td>
                  
                  <td>534</td>
                  
                  <td>64</td>
                  
                  <td>146690</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.214</td>
                  
                  <td>0.025</td>
                  
                  <td>0.034</td>
                  
                  <td>0.089</td>
                  
                  <td>0.06</td>
                  
                  <td>0.065</td>
                  
                  <td>0.164</td>
                  
                  <td>0.073</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.214,0.025,0.034,0.089,0.06,0.065,0.164,0.073]</td>
                  
                  <td>[0.05772141996750265,1.0068397668078486,0.1081511765981302]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200047</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>30</td>
                  
                  <td>36</td>
                  
                  <td>14</td>
                  
                  <td>32</td>
                  
                  <td>906</td>
                  
                  <td>61</td>
                  
                  <td>235668</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.375</td>
                  
                  <td>0.071</td>
                  
                  <td>0.089</td>
                  
                  <td>0.129</td>
                  
                  <td>0.111</td>
                  
                  <td>0.156</td>
                  
                  <td>0.117</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.0,0.375,0.071,0.089,0.129,0.111,0.156,0.117]</td>
                  
                  <td>[1.0363160712050736,0.9958417593885326,-0.20244905189905057]</td>
                  
                </tr>
                
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
  
  <script class="pd_save">
    $(function() {
      var tableWrapper = $('.df-table-wrapper-b3631c42');
      var fixedHeader = $('.fixed-header', tableWrapper);
      var tableContainer = $('.df-table-container', tableWrapper);
      var table = $('.df-table', tableContainer);
      var rows = $('tbody > tr', table);
      var total = 100;
  
      fixedHeader
        .css('width', table.width())
        .find('.fixed-cell')
        .each(function(i, e) {
          $(this).css('width', $('.df-table-wrapper-b3631c42 th:nth-child(' + (i+1) + ')').css('width'));
        });
  
      tableContainer.scroll(function() {
        fixedHeader.css({ left: table.position().left });
      });
  
      rows.on("click", function(e){
          var txt = e.delegateTarget.innerText;
          var splits = txt.split("\t");
          var len = splits.length;
          var hdrs = $(fixedHeader).find(".fixed-cell");
          // Add all cells in the selected row as a map to be consumed by the target as needed
          var payload = {type:"select", targetDivId: "" };
          for (var i = 0; i < len; i++) {
            payload[hdrs[i].innerHTML] = splits[i];
          }
  
          //simple selection highlighting, client adds "selected" class
          $(this).addClass("selected").siblings().removeClass("selected");
          $(document).trigger('pd_event', payload);
      });
  
      $('.df-table-search', tableWrapper).keyup(function() {
        var val = '^(?=.*\\b' + $.trim($(this).val()).split(/\s+/).join('\\b)(?=.*\\b') + ').*$';
        var reg = RegExp(val, 'i');
        var index = 0;
        
        rows.each(function(i, e) {
          if (!reg.test($(this).text().replace(/\s+/g, ' '))) {
            $(this).attr('class', 'hidden');
          }
          else {
            $(this).attr('class', (++index % 2 == 0 ? 'even' : 'odd'));
          }
        });
        $('.df-table-search-count', tableWrapper).html('Showing ' + index + ' of ' + total + ' rows');
      });
    });
  
    $(".df-table-wrapper td:contains('http://')").each(function(){var tc = this.textContent; $(this).wrapInner("<a target='_blank' href='" + tc + "'></a>");});
    $(".df-table-wrapper td:contains('https://')").each(function(){var tc = this.textContent; $(this).wrapInner("<a target='_blank' href='" + tc + "'></a>");});
  </script>
  
        </div>



```python
# Convert features_df to Pandas for further exploration
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

    /opt/ibm/conda/miniconda3.6/lib/python3.6/site-packages/ipykernel/__main__.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /opt/ibm/conda/miniconda3.6/lib/python3.6/site-packages/ipykernel/__main__.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


```python
# show relationship between variables for all features
plt.figure(figsize=(25, 25))
sns.pairplot(first_features_df, hue="label");
```

    /opt/ibm/conda/miniconda3.6/lib/python3.6/site-packages/statsmodels/nonparametric/kde.py:488: RuntimeWarning: invalid value encountered in true_divide
      binned = fast_linbin(X, a, b, gridsize) / (delta * nobs)
    /opt/ibm/conda/miniconda3.6/lib/python3.6/site-packages/statsmodels/nonparametric/kdetools.py:34: RuntimeWarning: invalid value encountered in double_scalars
      FAC1 = 2*(np.pi*bw/RANGE)**2
    /opt/ibm/conda/miniconda3.6/lib/python3.6/site-packages/numpy/core/fromnumeric.py:83: RuntimeWarning: invalid value encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    


    <Figure size 1800x1800 with 0 Axes>



![png](output_42_2.png)



```python
# show relationship between variables for all features
plt.figure(figsize=(25, 25))
sns.pairplot(second_features_df, hue="label");
```

    /opt/ibm/conda/miniconda3.6/lib/python3.6/site-packages/statsmodels/nonparametric/kde.py:488: RuntimeWarning: invalid value encountered in true_divide
      binned = fast_linbin(X, a, b, gridsize) / (delta * nobs)
    /opt/ibm/conda/miniconda3.6/lib/python3.6/site-packages/statsmodels/nonparametric/kdetools.py:34: RuntimeWarning: invalid value encountered in double_scalars
      FAC1 = 2*(np.pi*bw/RANGE)**2
    /opt/ibm/conda/miniconda3.6/lib/python3.6/site-packages/numpy/core/fromnumeric.py:83: RuntimeWarning: invalid value encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    


    <Figure size 1800x1800 with 0 Axes>



![png](output_43_2.png)


## Heat map on feature and label correlation


```python
# print correlation between variables
corr = pd_features_df.drop("userId", axis=1).corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True));
```


![png](output_45_0.png)


### Save model to csv to avoid long loading times for further model tuning ###


```python
# save to csv 
#project.save_data("features_unscaled_pd_df.csv", features_df.toPandas().to_csv(), overwrite = True)
```


```python
# load data from csv
#features_df = spark.read\
#  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
#  .option('header', 'true')\
#  .option("inferSchema", "true")\
#  .load(cos.url('features_unscaled_pd_df.csv', 'sparkify-donotdelete-pr-2exnp1jnopynlt'))
#features_df.take(3)
```

# Feature Selection

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
pca_features = model.transform(features_df.select("features")).select("pcaFeatures")

# join column "pcaFeatures" to existing dataframe
pca_features = pca_features.withColumn("id", F.monotonically_increasing_id())
features_df = features_df.withColumn("id", F.monotonically_increasing_id())
features_df = features_df.join(pca_features, "id", "outer").drop("id")

print("Explained variance by {} principal components: {:.2f}%".format(pca_number, sum(model.explainedVariance)*100))
```

    Explained variance by 3 principal components: 88.20%
    

## Save features as csv to avoid long loading times


```python
# save to csv 
#features_df.toPandas().to_csv("features_pd_df.csv")
#project.save_data("features_pd_df.csv", features_df.toPandas().to_csv(), overwrite = True)
```


```python
# load data from csv
#features_df = spark.read.load("features_pd_df.csv", format="csv", inferSchema="true", header="true")
#features_df = spark.read\
#  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
#  .option('header', 'true')\
#  .option("inferSchema", "true")\
#  .load(cos.url('features_pd_df.csv', 'sparkify-donotdelete-pr-2exnp1jnopynlt'))
#features_df.take(3)
```

# Modeling

## Split in training, test, validation set


```python
train, test = features_df.randomSplit([0.8, 0.2], seed=42)

plt.hist(features_df.toPandas()['label'])
plt.show()
```


![png](output_59_0.png)


## Analyze label class imbalance


```python
# calculate balancing ratio to account for class imbalance
balancing_ratio = train.filter(train['label']==0).count()/train.count()
train=train.withColumn("classWeights", F.when(train.label == 1,balancing_ratio).otherwise(1-balancing_ratio))
```

## Machine Learning Model Selection and Tuning

 * Model learning problem category: supervised learning, logistic regression
 * ML estimators from pyspark.ml:
     * LogisticRegression
 * ML hyperparameters in estimators (for grid search/ tuning):
     * LogisticRegression(maxIter=..., regParam=..., elasticNetParam=...)
 * ML evaluators from pyspark.ml:
     * BinaryClassificationEvaluator


```python
# Create a logistic regression object
lr_simple = LogisticRegression(featuresCol = 'pcaFeatures', labelCol = 'label', weightCol="classWeights")
```


```python
# fit training data to lr model and check performance before further refinement
lr_simple_model = lr_simple.fit(train)
training_summary = lr_simple_model.summary
```


```python
# print precision and recall
pr = training_summary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()
```


![png](output_65_0.png)



```python
# function that evaluates model prediction
def evaluate_prediction(df):
    '''
    ARGS: Spark Dateframe with columns "label" and "prediction"
    
    OUTPUT: F1_score, recall, prediction as float variables
    
    Function that prints F1 score, recall and prediction
    '''
    # evaluate results
    pd_pred = df.toPandas()

    # calculate score for f1, precision, recall
    f1 = f1_score(pd_pred.label, pd_pred.prediction)
    recall = recall_score(pd_pred.label, pd_pred.prediction)
    precision = precision_score(pd_pred.label, pd_pred.prediction)

    print("F1 Score: {:.2f}, Recall: {:.2f}, Precision: {:.2f}".format(f1, recall, precision))
    
    return f1, recall, precision
```


```python
# transform testing data and check results
simple_pred = lr_simple_model.transform(test)
evaluate_prediction(simple_pred)
```

    F1 Score: 0.30, Recall: 0.43, Precision: 0.23
    




    (0.3, 0.42857142857142855, 0.23076923076923078)




```python
lr_simple_model.extractParamMap()
```




    {Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='aggregationDepth', doc='suggested depth for treeAggregate (>= 2)'): 2,
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty'): 0.0,
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='family', doc='The name of family which is a description of the label distribution to be used in the model. Supported options: auto, binomial, multinomial.'): 'auto',
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='featuresCol', doc='features column name'): 'pcaFeatures',
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='fitIntercept', doc='whether to fit an intercept term'): True,
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='labelCol', doc='label column name'): 'label',
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='maxIter', doc='maximum number of iterations (>= 0)'): 100,
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='predictionCol', doc='prediction column name'): 'prediction',
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities'): 'probability',
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name'): 'rawPrediction',
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='regParam', doc='regularization parameter (>= 0)'): 0.0,
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='standardization', doc='whether to standardize the training features before fitting the model'): True,
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='threshold', doc='threshold in binary classification prediction, in range [0, 1]'): 0.5,
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='tol', doc='the convergence tolerance for iterative algorithms (>= 0)'): 1e-06,
     Param(parent='LogisticRegression_4d89a2ae9a7723a9d3b6', name='weightCol', doc='weight column name. If this is not set or empty, we treat all instance weights as 1.0'): 'classWeights'}




```python
# Create a logistic regression object
lr = LogisticRegression(featuresCol = 'pcaFeatures', labelCol = 'label', weightCol="classWeights")
```


```python
# create evaluator
evaluator = BinaryClassificationEvaluator(metricName = 'areaUnderPR')

# tune model via CrossValidator and parameter Grid 
# build paramGrid
lr_paramGrid = (ParamGridBuilder() \
    .addGrid(lr.maxIter, [1, 10, 100]) \
    .addGrid(lr.regParam,[0.0, 0.5, 2.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build())

# build cross validator
lr_crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=lr_paramGrid,
                          evaluator=evaluator,
                          numFolds=3)
```


```python
# run cross validation
# train model on train data
lr_crossval_model = lr_crossval.fit(train)
```


```python
# wait for 15 min = 900 seconds to avoid bug with prediction from trained spark model
time.sleep(900)
```

## Model Evaluation

* use scikit learn metrics f1, precision, recall for model evaluation


```python
# predict on test data
lr_crossval_pred = lr_crossval_model.transform(test)
```


```python
evaluate_prediction(lr_crossval_pred)
```

    F1 Score: 0.37, Recall: 0.62, Precision: 0.27
    




    (0.37142857142857144, 0.6190476190476191, 0.2653061224489796)



# Check Decision Tree as alternative estimator


```python
# Create a decision tree estimator object
dt = DecisionTreeClassifier(featuresCol = 'pcaFeatures', labelCol = 'label')
```


```python
# fit training data to lr model and check performance before further refinement
dt_simple_model = dt.fit(train)
#dt_training_summary = dt_model.summary
```


```python
# transform testing data and check results
dt_simple_pred = dt_simple_model.transform(test)
evaluate_prediction(dt_simple_pred)
```

    F1 Score: 0.40, Recall: 0.29, Precision: 0.67
    




    (0.4, 0.2857142857142857, 0.6666666666666666)




```python
# get parameters of simple model
dt_simple_model.extractParamMap()
```




    {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='cacheNodeIds', doc='If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees.'): False,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='checkpointInterval', doc='set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext'): 10,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='featuresCol', doc='features column name'): 'pcaFeatures',
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini',
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='labelCol', doc='label column name'): 'label',
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxMemoryInMB', doc='Maximum memory in MB allocated to histogram aggregation.'): 256,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='minInfoGain', doc='Minimum information gain for a split to be considered at a tree node.'): 0.0,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='minInstancesPerNode', doc='Minimum number of instances each child must have after split.  If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.'): 1,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='predictionCol', doc='prediction column name'): 'prediction',
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities'): 'probability',
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name'): 'rawPrediction',
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='seed', doc='random seed'): -233398890157416097}




```python
# tune dt model via CrossValidator and parameter Grid 
# build paramGrid
dt_paramGrid = (ParamGridBuilder() \
    .addGrid(dt.maxDepth, [2, 5, 8]) \
    .addGrid(dt.maxBins,[20, 32, 50]) \
    .addGrid(dt.impurity, ["gini", "entropy"]) \
    .build())

# build cross validator
dt_crossval = CrossValidator(estimator=dt,
                          estimatorParamMaps= dt_paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# run cross validation
# train model on train data
dt_crossval_model = dt_crossval.fit(train)

# wait for 15 min = 900 seconds to avoid bug with prediction from trained spark model
time.sleep(900)

# predict on test data
dt_crossval_pred = dt_crossval_model.transform(test)

#evaluate prediction results
evaluate_prediction(dt_crossval_pred)
```

    F1 Score: 0.25, Recall: 0.14, Precision: 1.00
    




    (0.25, 0.14285714285714285, 1.0)



## Get best model parameters ##
### Linear regression model parameters after tuning ###


```python
list(zip(lr_crossval_model.avgMetrics, lr_paramGrid))
```




    [(0.2258063798871866,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 1,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.0}),
     (0.2258063798871866,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 1,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.5}),
     (0.22580637988718658,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 1,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 1.0}),
     (0.2258063798871866,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 1,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.5,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.0}),
     (0.21617745352163462,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 1,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.5,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.5}),
     (0.21617745352163462,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 1,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.5,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 1.0}),
     (0.22580637988718658,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 1,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 2.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.0}),
     (0.21617745352163462,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 1,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 2.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.5}),
     (0.21617745352163462,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 1,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 2.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 1.0}),
     (0.22584689557919096,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 10,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.0}),
     (0.22584689557919096,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 10,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.5}),
     (0.22584689557919096,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 10,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 1.0}),
     (0.22548144044357699,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 10,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.5,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.0}),
     (0.21617745352163462,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 10,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.5,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.5}),
     (0.21617745352163462,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 10,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.5,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 1.0}),
     (0.22553746258411816,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 10,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 2.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.0}),
     (0.21617745352163462,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 10,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 2.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.5}),
     (0.21617745352163462,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 10,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 2.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 1.0}),
     (0.22584689557919096,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 100,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.0}),
     (0.22584689557919096,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 100,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.5}),
     (0.22584689557919096,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 100,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 1.0}),
     (0.22548144044357699,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 100,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.5,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.0}),
     (0.21617745352163462,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 100,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.5,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.5}),
     (0.21617745352163462,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 100,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 0.5,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 1.0}),
     (0.22553746258411816,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 100,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 2.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.0}),
     (0.21617745352163462,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 100,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 2.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.5}),
     (0.21617745352163462,
      {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='max number of iterations (>= 0).'): 100,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0).'): 2.0,
       Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 1.0})]




```python
best_lr_params = lr_crossval_model.bestModel.extractParamMap()
```


```python
best_lr_params
```




    {Param(parent='LogisticRegression_4a73acf1913266c719e1', name='aggregationDepth', doc='suggested depth for treeAggregate (>= 2)'): 2,
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty'): 0.0,
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='family', doc='The name of family which is a description of the label distribution to be used in the model. Supported options: auto, binomial, multinomial.'): 'auto',
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='featuresCol', doc='features column name'): 'pcaFeatures',
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='fitIntercept', doc='whether to fit an intercept term'): True,
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='labelCol', doc='label column name'): 'label',
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='maxIter', doc='maximum number of iterations (>= 0)'): 10,
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='predictionCol', doc='prediction column name'): 'prediction',
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities'): 'probability',
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name'): 'rawPrediction',
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='regParam', doc='regularization parameter (>= 0)'): 0.0,
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='standardization', doc='whether to standardize the training features before fitting the model'): True,
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='threshold', doc='threshold in binary classification prediction, in range [0, 1]'): 0.5,
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='tol', doc='the convergence tolerance for iterative algorithms (>= 0)'): 1e-06,
     Param(parent='LogisticRegression_4a73acf1913266c719e1', name='weightCol', doc='weight column name. If this is not set or empty, we treat all instance weights as 1.0'): 'classWeights'}



### Decision tree model parameters after tuning ###


```python
list(zip(dt_crossval_model.avgMetrics, dt_paramGrid))
```




    [(0.21209681713579853,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 2,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 20,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini'}),
     (0.20974292726182026,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 2,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 20,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'entropy'}),
     (0.1832455445850633,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 2,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini'}),
     (0.1831514972552981,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 2,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'entropy'}),
     (0.18311430998716238,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 2,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 50,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini'}),
     (0.18828594457279887,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 2,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 50,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'entropy'}),
     (0.2085351534879352,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 20,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini'}),
     (0.2036740187044527,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 20,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'entropy'}),
     (0.19291854035343206,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini'}),
     (0.21234327559124216,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'entropy'}),
     (0.2108058623526448,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 50,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini'}),
     (0.24159486099024807,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 50,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'entropy'}),
     (0.20869959354523807,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 8,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 20,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini'}),
     (0.23552448099390733,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 8,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 20,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'entropy'}),
     (0.18487915692764353,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 8,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini'}),
     (0.20522565438065526,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 8,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'entropy'}),
     (0.19726935236377735,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 8,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 50,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini'}),
     (0.1848453212636338,
      {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 8,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 50,
       Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'entropy'})]




```python
best_dt_params = dt_crossval_model.bestModel.extractParamMap()
```


```python
best_dt_params
```




    {Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='cacheNodeIds', doc='If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees.'): False,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='checkpointInterval', doc='set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext'): 10,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='featuresCol', doc='features column name'): 'pcaFeatures',
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'entropy',
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='labelCol', doc='label column name'): 'label',
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 50,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='maxMemoryInMB', doc='Maximum memory in MB allocated to histogram aggregation.'): 256,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='minInfoGain', doc='Minimum information gain for a split to be considered at a tree node.'): 0.0,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='minInstancesPerNode', doc='Minimum number of instances each child must have after split.  If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.'): 1,
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='predictionCol', doc='prediction column name'): 'prediction',
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities'): 'probability',
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name'): 'rawPrediction',
     Param(parent='DecisionTreeClassifier_42149583c31f07072aef', name='seed', doc='random seed'): -233398890157416097}




```python
end_time = datetime.datetime.now()
```


```python
print("start of notebook run at: ", str(start_time))
print("end of notebook run at: ", str(end_time))
```

    start of notebook run at:  2019-12-29 14:24:33.835321
    end of notebook run at:  2019-12-29 16:40:41.657971
    

## Save models to avoid long loading times ##
#lr_simple
from pyspark.ml import Pipeline
pipeline_org = Pipeline( stages=[ lr_simple ] )
pipeline_model_org = pipeline_org.fit(train)
pipeline_model_org.save( "tent-prediction-model" )
#train_org.write.save( "training-data.parquet" )

# Load the model and training data into memory
from pyspark.ml import PipelineModel
pipeline_model = PipelineModel.load("tent-prediction-model")
pipeline = Pipeline( stages = pipeline_model.stages )
#train = spark.read.load( "training-data.parquet" )# models:
# lr_simple_model
# lr_crossval_model
# dt_simple_model
# dt_crossval_model
#project.save_data("lr_simple_model", lr_simple_model.save("lr_simple_model"))
#project.save_data("lr_crossval_model", lr_crossval_model.save())
#project.save_data("dt_simple_model", dt_simple_model.save())
#project.save_data("dt_crossval_model", dt_crossval_model.save())

# Store the model
from watson_machine_learning_client import WatsonMachineLearningAPIClient

wml_credentials={
  "apikey": "XONLozlb_YwufFQASFd8j3RwFlvKxOa1UPzbX6PF9_wq",
  "iam_apikey_description": "Auto generated apikey during resource-key operation for Instance - crn:v1:bluemix:public:pm-20:eu-gb:a/a6e585a0e8bf4e49860d08610a6aafcb:96dc036b-4823-4a43-82f7-1b0f385b4b63::",
  "iam_apikey_name": "auto-generated-apikey-729ad668-d97d-4982-9e4e-8857c0399752",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/a6e585a0e8bf4e49860d08610a6aafcb::serviceid:ServiceId-f92cad2e-dfc4-460b-b6da-6823a8a1941c",
  "instance_id": "96dc036b-4823-4a43-82f7-1b0f385b4b63",
  "url": "https://eu-gb.ml.cloud.ibm.com"
}

client = WatsonMachineLearningAPIClient(wml_credentials)

model_details = client.repository.store_model( pipeline_model, 'simple_lr_model', training_data=train, pipeline=pipeline )#Specify the Properties 
model_props = {client.repository.ModelMetaNames.AUTHOR_NAME: "Stefan", 
               client.repository.ModelMetaNames.AUTHOR_EMAIL: "", 
               client.repository.ModelMetaNames.NAME: "lr_simple_model"
              }
#Store the Machine Learning Model
model_artifact=client.repository.store_model(lr_simple_model, meta_props=model_props)client.repository.list()