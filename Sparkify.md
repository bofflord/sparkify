
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
```

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
            <span>Pixiedust version 1.1.16</span>
        </div>
        


    [31mPixiedust runtime updated. Please restart kernel[0m
    Table SPARK_PACKAGES created successfully
    Table USER_PREFERENCES created successfully
    Table service_connections created successfully



<div>Warning: You are not running the latest version of PixieDust. Current is 1.1.16, Latest is 1.1.17</div>




                <div>Please copy and run the following command in a new cell to upgrade: <span style="background-color:#ececec;font-family:monospace;padding:0 5px">!pip install --user --upgrade pixiedust</span></div>
            



<div>Please restart kernel after upgrading.</div>



```python
pixiedust.optOut()
```

    Pixiedust will not collect anonymous install statistics.


# Explore and clean data set


```python
# Peek at dataset
display(user_log)
```


<style type="text/css">.pd_warning{display:none;}</style><div class="pd_warning"><em>Hey, there's something awesome here! To see it, open this notebook outside GitHub, in a viewer like Jupyter</em></div>
        <div class="pd_save is-viewer-good" style="padding-right:10px;text-align: center;line-height:initial !important;font-size: xx-large;font-weight: 500;color: coral;">
            
        </div>
    <div id="chartFiguref0f0f373" class="pd_save is-viewer-good" style="overflow-x:auto">
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
  
  
  <div class="df-table-wrapper df-table-wrapper-f0f0f373 panel-group pd_save">
    <!-- dataframe schema -->
    
    <div class="panel panel-default">
      <div class="panel-heading">
        <h4 class="panel-title" style="margin: 0px;">
          <a data-toggle="collapse" href="#df-schema-f0f0f373" data-parent="#df-table-wrapper-f0f0f373">Schema</a>
        </h4>
      </div>
      <div id="df-schema-f0f0f373" class="panel-collapse collapse">
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
          <a data-toggle="collapse" href="#df-table-f0f0f373" data-parent="#df-table-wrapper-f0f0f373"> Table</a>
        </h4>
      </div>
      
      <div id="df-table-f0f0f373" class="panel-collapse collapse in">
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
                  
                  <td>Colin</td>
                  
                  <td>M</td>
                  
                  <td>12</td>
                  
                  <td>Larson</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1537982255000</td>
                  
                  <td>497</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538353849000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0</td>
                  
                  <td>100</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 00:30:49</td>
                  
                  <td>0</td>
                  
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
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>6</td>
                  
                  <td>Taylor</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1533764798000</td>
                  
                  <td>195</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538354331000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
                  <td>0</td>
                  
                  <td>VA-NC</td>
                  
                  <td>2018-10-01 00:38:51</td>
                  
                  <td>0</td>
                  
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
                  
                  <td>Postal Service</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>13</td>
                  
                  <td>Taylor</td>
                  
                  <td>226.61179</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533764798000</td>
                  
                  <td>195</td>
                  
                  <td>Nothing Better (Album)</td>
                  
                  <td>200</td>
                  
                  <td>1538355538000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
                  <td>0</td>
                  
                  <td>VA-NC</td>
                  
                  <td>2018-10-01 00:58:58</td>
                  
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
                  
                  <td>The Chills</td>
                  
                  <td>Logged In</td>
                  
                  <td>Payton</td>
                  
                  <td>F</td>
                  
                  <td>104</td>
                  
                  <td>Campbell</td>
                  
                  <td>237.76608</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529027541000</td>
                  
                  <td>38</td>
                  
                  <td>Pink Frost</td>
                  
                  <td>200</td>
                  
                  <td>1538355917000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"</td>
                  
                  <td>39</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 01:05:17</td>
                  
                  <td>1</td>
                  
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
                  
                  <td>Flying Lotus</td>
                  
                  <td>Logged In</td>
                  
                  <td>Colin</td>
                  
                  <td>M</td>
                  
                  <td>20</td>
                  
                  <td>Larson</td>
                  
                  <td>178.25914</td>
                  
                  <td>free</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537982255000</td>
                  
                  <td>497</td>
                  
                  <td>Orbit Brazil</td>
                  
                  <td>200</td>
                  
                  <td>1538355989000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0</td>
                  
                  <td>100</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 01:06:29</td>
                  
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
                  
                  <td>Comisarios de la Sierra</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>8</td>
                  
                  <td>Hogan</td>
                  
                  <td>147.69587</td>
                  
                  <td>free</td>
                  
                  <td>Denver-Aurora-Lakewood, CO</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535066380000</td>
                  
                  <td>100</td>
                  
                  <td>El Corrido De Julian</td>
                  
                  <td>200</td>
                  
                  <td>1538360261000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.2; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>101</td>
                  
                  <td>0</td>
                  
                  <td>CO</td>
                  
                  <td>2018-10-01 02:17:41</td>
                  
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
                  
                  <td>HYPOCRISY</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>41</td>
                  
                  <td>Taylor</td>
                  
                  <td>263.13098</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533764798000</td>
                  
                  <td>195</td>
                  
                  <td>All turns black</td>
                  
                  <td>200</td>
                  
                  <td>1538360693000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
                  <td>0</td>
                  
                  <td>VA-NC</td>
                  
                  <td>2018-10-01 02:24:53</td>
                  
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
                  
                  <td>Michael BublÃÂ©</td>
                  
                  <td>Logged In</td>
                  
                  <td>Carter</td>
                  
                  <td>M</td>
                  
                  <td>9</td>
                  
                  <td>Cook</td>
                  
                  <td>576.83546</td>
                  
                  <td>paid</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1522793334000</td>
                  
                  <td>287</td>
                  
                  <td>Save The Last Dance For Me [Ralphi Rosario Anthomic Vocal]</td>
                  
                  <td>200</td>
                  
                  <td>1538361107000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>288</td>
                  
                  <td>0</td>
                  
                  <td>IL-IN-WI</td>
                  
                  <td>2018-10-01 02:31:47</td>
                  
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
                  
                  <td>Kamelot</td>
                  
                  <td>Logged In</td>
                  
                  <td>Sofia</td>
                  
                  <td>F</td>
                  
                  <td>311</td>
                  
                  <td>Gordon</td>
                  
                  <td>253.23057</td>
                  
                  <td>paid</td>
                  
                  <td>Rochester, MN</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533175710000</td>
                  
                  <td>162</td>
                  
                  <td>EdenEcho</td>
                  
                  <td>200</td>
                  
                  <td>1538362153000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>163</td>
                  
                  <td>0</td>
                  
                  <td>MN</td>
                  
                  <td>2018-10-01 02:49:13</td>
                  
                  <td>2</td>
                  
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
                  
                  <td>Sticky Fingaz / Still Living / X-1</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lucas</td>
                  
                  <td>M</td>
                  
                  <td>32</td>
                  
                  <td>Decker</td>
                  
                  <td>297.482</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534945722000</td>
                  
                  <td>222</td>
                  
                  <td>Why</td>
                  
                  <td>200</td>
                  
                  <td>1538362476000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>223</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 02:54:36</td>
                  
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
                  
                  <td>The New Pornographers</td>
                  
                  <td>Logged In</td>
                  
                  <td>Maverick</td>
                  
                  <td>M</td>
                  
                  <td>240</td>
                  
                  <td>Santiago</td>
                  
                  <td>257.38404</td>
                  
                  <td>paid</td>
                  
                  <td>Orlando-Kissimmee-Sanford, FL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535953455000</td>
                  
                  <td>178</td>
                  
                  <td>My Rights Versus Yours</td>
                  
                  <td>200</td>
                  
                  <td>1538363786000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>179</td>
                  
                  <td>0</td>
                  
                  <td>FL</td>
                  
                  <td>2018-10-01 03:16:26</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>McFly</td>
                  
                  <td>Logged In</td>
                  
                  <td>Aurora</td>
                  
                  <td>F</td>
                  
                  <td>5</td>
                  
                  <td>Humphrey</td>
                  
                  <td>215.92771</td>
                  
                  <td>paid</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536795126000</td>
                  
                  <td>418</td>
                  
                  <td>The Way You Make Me Feel</td>
                  
                  <td>200</td>
                  
                  <td>1538364163000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>127</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 03:22:43</td>
                  
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
                  
                  <td>The Weepies</td>
                  
                  <td>Logged In</td>
                  
                  <td>Carter</td>
                  
                  <td>M</td>
                  
                  <td>32</td>
                  
                  <td>Cook</td>
                  
                  <td>163.18649</td>
                  
                  <td>paid</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1522793334000</td>
                  
                  <td>287</td>
                  
                  <td>World Spins Madly On</td>
                  
                  <td>200</td>
                  
                  <td>1538366382000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>288</td>
                  
                  <td>0</td>
                  
                  <td>IL-IN-WI</td>
                  
                  <td>2018-10-01 03:59:42</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Blitzen Trapper</td>
                  
                  <td>Logged In</td>
                  
                  <td>Maverick</td>
                  
                  <td>M</td>
                  
                  <td>257</td>
                  
                  <td>Santiago</td>
                  
                  <td>171.20608</td>
                  
                  <td>paid</td>
                  
                  <td>Orlando-Kissimmee-Sanford, FL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535953455000</td>
                  
                  <td>178</td>
                  
                  <td>Fire &amp; Fast Bullets</td>
                  
                  <td>200</td>
                  
                  <td>1538367329000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>179</td>
                  
                  <td>0</td>
                  
                  <td>FL</td>
                  
                  <td>2018-10-01 04:15:29</td>
                  
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
                  
                  <td>Survivor</td>
                  
                  <td>Logged In</td>
                  
                  <td>Caleb</td>
                  
                  <td>M</td>
                  
                  <td>15</td>
                  
                  <td>Lane</td>
                  
                  <td>245.36771</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536756625000</td>
                  
                  <td>281</td>
                  
                  <td>Eye Of The Tiger</td>
                  
                  <td>200</td>
                  
                  <td>1538367341000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>282</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 04:15:41</td>
                  
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
                  
                  <td>OneRepublic</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jhaden</td>
                  
                  <td>M</td>
                  
                  <td>28</td>
                  
                  <td>Cooper</td>
                  
                  <td>224.67873</td>
                  
                  <td>free</td>
                  
                  <td>Knoxville, TN</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536859413000</td>
                  
                  <td>249</td>
                  
                  <td>Secrets</td>
                  
                  <td>200</td>
                  
                  <td>1538367412000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>250</td>
                  
                  <td>0</td>
                  
                  <td>TN</td>
                  
                  <td>2018-10-01 04:16:52</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Caleb</td>
                  
                  <td>M</td>
                  
                  <td>17</td>
                  
                  <td>Lane</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1536756625000</td>
                  
                  <td>281</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538367598000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>282</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 04:19:58</td>
                  
                  <td>4</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Dillyn</td>
                  
                  <td>F</td>
                  
                  <td>1</td>
                  
                  <td>Richardson</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Danville, VA</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1537811988000</td>
                  
                  <td>478</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538367818000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>42</td>
                  
                  <td>0</td>
                  
                  <td>VA</td>
                  
                  <td>2018-10-01 04:23:38</td>
                  
                  <td>4</td>
                  
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
                  
                  <td>Relient K</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>33</td>
                  
                  <td>Roberts</td>
                  
                  <td>239.90812</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537634865000</td>
                  
                  <td>27</td>
                  
                  <td>Be My Escape (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538368292000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                  <td>0</td>
                  
                  <td>OH</td>
                  
                  <td>2018-10-01 04:31:32</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Oliver</td>
                  
                  <td>M</td>
                  
                  <td>5</td>
                  
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
                  
                  <td>1538368443000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>154</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 04:34:03</td>
                  
                  <td>4</td>
                  
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
                  
                  <td>Maverick</td>
                  
                  <td>M</td>
                  
                  <td>270</td>
                  
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
                  
                  <td>1538369208000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>179</td>
                  
                  <td>0</td>
                  
                  <td>FL</td>
                  
                  <td>2018-10-01 04:46:48</td>
                  
                  <td>4</td>
                  
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
                  
                  <td>Nine Inch Nails</td>
                  
                  <td>Logged In</td>
                  
                  <td>Carter</td>
                  
                  <td>M</td>
                  
                  <td>52</td>
                  
                  <td>Cook</td>
                  
                  <td>273.21424</td>
                  
                  <td>paid</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1522793334000</td>
                  
                  <td>287</td>
                  
                  <td>Head Like A Hole</td>
                  
                  <td>200</td>
                  
                  <td>1538369750000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>288</td>
                  
                  <td>0</td>
                  
                  <td>IL-IN-WI</td>
                  
                  <td>2018-10-01 04:55:50</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Anthony</td>
                  
                  <td>M</td>
                  
                  <td>61</td>
                  
                  <td>Reed</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Miami-Fort Lauderdale-West Palm Beach, FL</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1534823030000</td>
                  
                  <td>511</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538370366000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>166</td>
                  
                  <td>0</td>
                  
                  <td>FL</td>
                  
                  <td>2018-10-01 05:06:06</td>
                  
                  <td>5</td>
                  
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
                  
                  <td>Van Halen</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>54</td>
                  
                  <td>Roberts</td>
                  
                  <td>311.61424</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537634865000</td>
                  
                  <td>27</td>
                  
                  <td>Love Walks In (Remastered Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538371226000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                  <td>0</td>
                  
                  <td>OH</td>
                  
                  <td>2018-10-01 05:20:26</td>
                  
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
                  
                  <td>The Avett Brothers</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kael</td>
                  
                  <td>M</td>
                  
                  <td>31</td>
                  
                  <td>Baker</td>
                  
                  <td>125.33506</td>
                  
                  <td>free</td>
                  
                  <td>Kingsport-Bristol-Bristol, TN-VA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533102330000</td>
                  
                  <td>487</td>
                  
                  <td>Hard Worker</td>
                  
                  <td>200</td>
                  
                  <td>1538371363000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"</td>
                  
                  <td>131</td>
                  
                  <td>0</td>
                  
                  <td>TN-VA</td>
                  
                  <td>2018-10-01 05:22:43</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lucas</td>
                  
                  <td>M</td>
                  
                  <td>76</td>
                  
                  <td>Decker</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1534945722000</td>
                  
                  <td>222</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538371547000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>223</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 05:25:47</td>
                  
                  <td>5</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Gustavo Santaolalla</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kael</td>
                  
                  <td>M</td>
                  
                  <td>33</td>
                  
                  <td>Baker</td>
                  
                  <td>91.89832</td>
                  
                  <td>free</td>
                  
                  <td>Kingsport-Bristol-Bristol, TN-VA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533102330000</td>
                  
                  <td>487</td>
                  
                  <td>Walking In Tokyo</td>
                  
                  <td>200</td>
                  
                  <td>1538371879000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"</td>
                  
                  <td>131</td>
                  
                  <td>0</td>
                  
                  <td>TN-VA</td>
                  
                  <td>2018-10-01 05:31:19</td>
                  
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
                  
                  <td>Fleet Foxes</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lucas</td>
                  
                  <td>M</td>
                  
                  <td>87</td>
                  
                  <td>Decker</td>
                  
                  <td>147.04281</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534945722000</td>
                  
                  <td>222</td>
                  
                  <td>White Winter Hymnal</td>
                  
                  <td>200</td>
                  
                  <td>1538373104000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>223</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 05:51:44</td>
                  
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
                  
                  <td>Armin van Buuren</td>
                  
                  <td>Logged In</td>
                  
                  <td>Adriel</td>
                  
                  <td>M</td>
                  
                  <td>20</td>
                  
                  <td>Mendoza</td>
                  
                  <td>435.51302</td>
                  
                  <td>paid</td>
                  
                  <td>Kansas City, MO-KS</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535623466000</td>
                  
                  <td>476</td>
                  
                  <td>Hold On To Me</td>
                  
                  <td>200</td>
                  
                  <td>1538373704000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"</td>
                  
                  <td>18</td>
                  
                  <td>0</td>
                  
                  <td>MO-KS</td>
                  
                  <td>2018-10-01 06:01:44</td>
                  
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
                  
                  <td>Nora</td>
                  
                  <td>F</td>
                  
                  <td>6</td>
                  
                  <td>Kennedy</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Madison, WI</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1533944611000</td>
                  
                  <td>524</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538374326000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>301</td>
                  
                  <td>0</td>
                  
                  <td>WI</td>
                  
                  <td>2018-10-01 06:12:06</td>
                  
                  <td>6</td>
                  
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
                  
                  <td>Young Money / Gucci Mane</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>143</td>
                  
                  <td>Campos</td>
                  
                  <td>310.72608</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000</td>
                  
                  <td>245</td>
                  
                  <td>Steady Mobbin</td>
                  
                  <td>200</td>
                  
                  <td>1538375154000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                  <td>0</td>
                  
                  <td>AL</td>
                  
                  <td>2018-10-01 06:25:54</td>
                  
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
                  
                  <td>Reality Check</td>
                  
                  <td>Logged In</td>
                  
                  <td>Oliver</td>
                  
                  <td>M</td>
                  
                  <td>39</td>
                  
                  <td>Fry</td>
                  
                  <td>226.19383</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538048434000</td>
                  
                  <td>153</td>
                  
                  <td>Masquerade (Reality Check Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538375253000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>154</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 06:27:33</td>
                  
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
                  
                  <td>La Mancha De Rolando</td>
                  
                  <td>Logged In</td>
                  
                  <td>Oliver</td>
                  
                  <td>M</td>
                  
                  <td>50</td>
                  
                  <td>Fry</td>
                  
                  <td>243.98322</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538048434000</td>
                  
                  <td>153</td>
                  
                  <td>Melodia Simple</td>
                  
                  <td>200</td>
                  
                  <td>1538377141000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>154</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 06:59:01</td>
                  
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
                  
                  <td>Silversun Pickups</td>
                  
                  <td>Logged In</td>
                  
                  <td>Carter</td>
                  
                  <td>M</td>
                  
                  <td>101</td>
                  
                  <td>Cook</td>
                  
                  <td>348.23791</td>
                  
                  <td>paid</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1522793334000</td>
                  
                  <td>287</td>
                  
                  <td>Lazy Eye [Jason Bentley Remix]</td>
                  
                  <td>200</td>
                  
                  <td>1538379700000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>288</td>
                  
                  <td>0</td>
                  
                  <td>IL-IN-WI</td>
                  
                  <td>2018-10-01 07:41:40</td>
                  
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
                  
                  <td>The Ruts</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>30</td>
                  
                  <td>Johnson</td>
                  
                  <td>338.96444</td>
                  
                  <td>paid</td>
                  
                  <td>Lexington-Fayette, KY</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538080987000</td>
                  
                  <td>493</td>
                  
                  <td>West One (Shine On Me)</td>
                  
                  <td>200</td>
                  
                  <td>1538380154000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>51</td>
                  
                  <td>0</td>
                  
                  <td>KY</td>
                  
                  <td>2018-10-01 07:49:14</td>
                  
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
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>0</td>
                  
                  <td>Taylor</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1533764798000</td>
                  
                  <td>522</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538380223000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
                  <td>0</td>
                  
                  <td>VA-NC</td>
                  
                  <td>2018-10-01 07:50:23</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Caifanes</td>
                  
                  <td>Logged In</td>
                  
                  <td>Carter</td>
                  
                  <td>M</td>
                  
                  <td>106</td>
                  
                  <td>Cook</td>
                  
                  <td>279.17016</td>
                  
                  <td>paid</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1522793334000</td>
                  
                  <td>287</td>
                  
                  <td>No Dejes Que...</td>
                  
                  <td>200</td>
                  
                  <td>1538380491000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>288</td>
                  
                  <td>0</td>
                  
                  <td>IL-IN-WI</td>
                  
                  <td>2018-10-01 07:54:51</td>
                  
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
                  
                  <td>MGMT</td>
                  
                  <td>Logged In</td>
                  
                  <td>Faigy</td>
                  
                  <td>F</td>
                  
                  <td>46</td>
                  
                  <td>Howe</td>
                  
                  <td>149.57669</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538211832000</td>
                  
                  <td>492</td>
                  
                  <td>Someone's Missing</td>
                  
                  <td>200</td>
                  
                  <td>1538382404000</td>
                  
                  <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>95</td>
                  
                  <td>0</td>
                  
                  <td>AZ</td>
                  
                  <td>2018-10-01 08:26:44</td>
                  
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
                  
                  <td>Gangster Fun</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nora</td>
                  
                  <td>F</td>
                  
                  <td>47</td>
                  
                  <td>Kennedy</td>
                  
                  <td>195.91791</td>
                  
                  <td>free</td>
                  
                  <td>Madison, WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533944611000</td>
                  
                  <td>524</td>
                  
                  <td>Skarabia</td>
                  
                  <td>200</td>
                  
                  <td>1538384145000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>301</td>
                  
                  <td>0</td>
                  
                  <td>WI</td>
                  
                  <td>2018-10-01 08:55:45</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Mischa Daniels Feat. Tash</td>
                  
                  <td>Logged In</td>
                  
                  <td>Daelin</td>
                  
                  <td>M</td>
                  
                  <td>2</td>
                  
                  <td>Turner</td>
                  
                  <td>312.81587</td>
                  
                  <td>free</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538227408000</td>
                  
                  <td>125</td>
                  
                  <td>Round &amp; Round (Take Me Higher)</td>
                  
                  <td>200</td>
                  
                  <td>1538387318000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>126</td>
                  
                  <td>0</td>
                  
                  <td>NY-NJ-PA</td>
                  
                  <td>2018-10-01 09:48:38</td>
                  
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
                  
                  <td>Harmonia</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>143</td>
                  
                  <td>Roberts</td>
                  
                  <td>655.77751</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537634865000</td>
                  
                  <td>27</td>
                  
                  <td>Sehr kosmisch</td>
                  
                  <td>200</td>
                  
                  <td>1538389869000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                  <td>0</td>
                  
                  <td>OH</td>
                  
                  <td>2018-10-01 10:31:09</td>
                  
                  <td>10</td>
                  
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
                  
                  <td>Silversun Pickups</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>27</td>
                  
                  <td>Myers</td>
                  
                  <td>265.7171</td>
                  
                  <td>paid</td>
                  
                  <td>Grand Rapids-Wyoming, MI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529995579000</td>
                  
                  <td>235</td>
                  
                  <td>Booksmart Devil</td>
                  
                  <td>200</td>
                  
                  <td>1538391689000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>236</td>
                  
                  <td>0</td>
                  
                  <td>MI</td>
                  
                  <td>2018-10-01 11:01:29</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Trance Atlantic Air Waves</td>
                  
                  <td>Logged In</td>
                  
                  <td>Emily</td>
                  
                  <td>F</td>
                  
                  <td>18</td>
                  
                  <td>Williams</td>
                  
                  <td>299.15383</td>
                  
                  <td>free</td>
                  
                  <td>Austin-Round Rock, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536852701000</td>
                  
                  <td>172</td>
                  
                  <td>Addiction Day</td>
                  
                  <td>200</td>
                  
                  <td>1538391919000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>173</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 11:05:19</td>
                  
                  <td>11</td>
                  
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
                  
                  <td>Benny Benassi Presents The Biz</td>
                  
                  <td>Logged In</td>
                  
                  <td>Evan</td>
                  
                  <td>M</td>
                  
                  <td>37</td>
                  
                  <td>Shelton</td>
                  
                  <td>285.54404</td>
                  
                  <td>free</td>
                  
                  <td>Hagerstown-Martinsburg, MD-WV</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534894284000</td>
                  
                  <td>479</td>
                  
                  <td>Satisfaction</td>
                  
                  <td>200</td>
                  
                  <td>1538393029000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>251</td>
                  
                  <td>0</td>
                  
                  <td>MD-WV</td>
                  
                  <td>2018-10-01 11:23:49</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>163</td>
                  
                  <td>Roberts</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1537634865000</td>
                  
                  <td>27</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538393218000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                  <td>0</td>
                  
                  <td>OH</td>
                  
                  <td>2018-10-01 11:26:58</td>
                  
                  <td>11</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Emily</td>
                  
                  <td>F</td>
                  
                  <td>26</td>
                  
                  <td>Williams</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Austin-Round Rock, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1536852701000</td>
                  
                  <td>172</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538393321000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>173</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 11:28:41</td>
                  
                  <td>11</td>
                  
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
                  
                  <td>The Starting Line</td>
                  
                  <td>Logged In</td>
                  
                  <td>Daelin</td>
                  
                  <td>M</td>
                  
                  <td>40</td>
                  
                  <td>Turner</td>
                  
                  <td>218.5922</td>
                  
                  <td>free</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538227408000</td>
                  
                  <td>125</td>
                  
                  <td>Up And Go</td>
                  
                  <td>200</td>
                  
                  <td>1538394324000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>126</td>
                  
                  <td>0</td>
                  
                  <td>NY-NJ-PA</td>
                  
                  <td>2018-10-01 11:45:24</td>
                  
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
                  
                  <td>Katy Perry</td>
                  
                  <td>Logged In</td>
                  
                  <td>Brayden</td>
                  
                  <td>M</td>
                  
                  <td>47</td>
                  
                  <td>Thomas</td>
                  
                  <td>255.55546</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534133898000</td>
                  
                  <td>498</td>
                  
                  <td>Lost</td>
                  
                  <td>200</td>
                  
                  <td>1538394895000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>85</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 11:54:55</td>
                  
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
                  
                  <td>Base Ball Bear</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>170</td>
                  
                  <td>Roberts</td>
                  
                  <td>255.60771</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537634865000</td>
                  
                  <td>27</td>
                  
                  <td>Sayonara-Nostalgia</td>
                  
                  <td>200</td>
                  
                  <td>1538395229000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                  <td>0</td>
                  
                  <td>OH</td>
                  
                  <td>2018-10-01 12:00:29</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Olivia</td>
                  
                  <td>F</td>
                  
                  <td>15</td>
                  
                  <td>Carr</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Fort Wayne, IN</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1536758439000</td>
                  
                  <td>490</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538395948000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.2; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>208</td>
                  
                  <td>0</td>
                  
                  <td>IN</td>
                  
                  <td>2018-10-01 12:12:28</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Colossal</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>4</td>
                  
                  <td>Garrett</td>
                  
                  <td>206.99383</td>
                  
                  <td>free</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537385168000</td>
                  
                  <td>186</td>
                  
                  <td>Brave The Elements</td>
                  
                  <td>200</td>
                  
                  <td>1538396818000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>187</td>
                  
                  <td>0</td>
                  
                  <td>IL-IN-WI</td>
                  
                  <td>2018-10-01 12:26:58</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Pearl Jam</td>
                  
                  <td>Logged In</td>
                  
                  <td>Riley</td>
                  
                  <td>F</td>
                  
                  <td>0</td>
                  
                  <td>Taylor</td>
                  
                  <td>283.89832</td>
                  
                  <td>free</td>
                  
                  <td>Boston-Cambridge-Newton, MA-NH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536403972000</td>
                  
                  <td>91</td>
                  
                  <td>Rearviewmirror</td>
                  
                  <td>200</td>
                  
                  <td>1538397670000</td>
                  
                  <td>"Mozilla/5.0 (iPad; CPU OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>92</td>
                  
                  <td>0</td>
                  
                  <td>MA-NH</td>
                  
                  <td>2018-10-01 12:41:10</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Tommy James And The Shondells</td>
                  
                  <td>Logged In</td>
                  
                  <td>Payton</td>
                  
                  <td>F</td>
                  
                  <td>30</td>
                  
                  <td>Campbell</td>
                  
                  <td>173.73995</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529027541000</td>
                  
                  <td>518</td>
                  
                  <td>Hanky Panky (Mono)</td>
                  
                  <td>200</td>
                  
                  <td>1538397836000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"</td>
                  
                  <td>39</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 12:43:56</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>5</td>
                  
                  <td>Taylor</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>PUT</td>
                  
                  <td>Add Friend</td>
                  
                  <td>1533764798000</td>
                  
                  <td>547</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538398541000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
                  <td>0</td>
                  
                  <td>VA-NC</td>
                  
                  <td>2018-10-01 12:55:41</td>
                  
                  <td>12</td>
                  
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
                  
                  <td>Essential Logic</td>
                  
                  <td>Logged In</td>
                  
                  <td>Olivia</td>
                  
                  <td>F</td>
                  
                  <td>32</td>
                  
                  <td>Carr</td>
                  
                  <td>203.67628</td>
                  
                  <td>free</td>
                  
                  <td>Fort Wayne, IN</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536758439000</td>
                  
                  <td>490</td>
                  
                  <td>Do You Believe In Christmas?</td>
                  
                  <td>200</td>
                  
                  <td>1538399599000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.2; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>208</td>
                  
                  <td>0</td>
                  
                  <td>IN</td>
                  
                  <td>2018-10-01 13:13:19</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Twista</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>75</td>
                  
                  <td>Myers</td>
                  
                  <td>223.03302</td>
                  
                  <td>paid</td>
                  
                  <td>Grand Rapids-Wyoming, MI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529995579000</td>
                  
                  <td>235</td>
                  
                  <td>Girl Tonite [Featuring Trey Songz] [Explicit Album Version]</td>
                  
                  <td>200</td>
                  
                  <td>1538402203000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>236</td>
                  
                  <td>0</td>
                  
                  <td>MI</td>
                  
                  <td>2018-10-01 13:56:43</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Bunbury</td>
                  
                  <td>Logged In</td>
                  
                  <td>Faigy</td>
                  
                  <td>F</td>
                  
                  <td>149</td>
                  
                  <td>Howe</td>
                  
                  <td>221.1522</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538211832000</td>
                  
                  <td>492</td>
                  
                  <td>Watching The Wheels</td>
                  
                  <td>200</td>
                  
                  <td>1538403143000</td>
                  
                  <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>95</td>
                  
                  <td>0</td>
                  
                  <td>AZ</td>
                  
                  <td>2018-10-01 14:12:23</td>
                  
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
                  
                  <td>Foo Fighters</td>
                  
                  <td>Logged In</td>
                  
                  <td>Anna</td>
                  
                  <td>F</td>
                  
                  <td>7</td>
                  
                  <td>Williams</td>
                  
                  <td>250.14812</td>
                  
                  <td>free</td>
                  
                  <td>Charlotte-Concord-Gastonia, NC-SC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1526838391000</td>
                  
                  <td>425</td>
                  
                  <td>Everlong</td>
                  
                  <td>200</td>
                  
                  <td>1538403796000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>119</td>
                  
                  <td>0</td>
                  
                  <td>NC-SC</td>
                  
                  <td>2018-10-01 14:23:16</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Molly</td>
                  
                  <td>F</td>
                  
                  <td>10</td>
                  
                  <td>Harrison</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1534255113000</td>
                  
                  <td>142</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538403893000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>143</td>
                  
                  <td>0</td>
                  
                  <td>VA-NC</td>
                  
                  <td>2018-10-01 14:24:53</td>
                  
                  <td>14</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Anna</td>
                  
                  <td>F</td>
                  
                  <td>17</td>
                  
                  <td>Williams</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Charlotte-Concord-Gastonia, NC-SC</td>
                  
                  <td>GET</td>
                  
                  <td>Upgrade</td>
                  
                  <td>1526838391000</td>
                  
                  <td>425</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538404336000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>119</td>
                  
                  <td>0</td>
                  
                  <td>NC-SC</td>
                  
                  <td>2018-10-01 14:32:16</td>
                  
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
                  
                  <td>Jamie Foxx</td>
                  
                  <td>Logged In</td>
                  
                  <td>Anna</td>
                  
                  <td>F</td>
                  
                  <td>18</td>
                  
                  <td>Williams</td>
                  
                  <td>239.15057</td>
                  
                  <td>free</td>
                  
                  <td>Charlotte-Concord-Gastonia, NC-SC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1526838391000</td>
                  
                  <td>425</td>
                  
                  <td>Any Given Sunday [feat. Guru &amp; Common] (Explicit Soundtrack Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538404365000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>119</td>
                  
                  <td>0</td>
                  
                  <td>NC-SC</td>
                  
                  <td>2018-10-01 14:32:45</td>
                  
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
                  
                  <td>Wilco</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>27</td>
                  
                  <td>Raymond</td>
                  
                  <td>220.76036</td>
                  
                  <td>free</td>
                  
                  <td>Hartford-West Hartford-East Hartford, CT</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534245996000</td>
                  
                  <td>26</td>
                  
                  <td>One Wing</td>
                  
                  <td>200</td>
                  
                  <td>1538405476000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                  <td>0</td>
                  
                  <td>CT</td>
                  
                  <td>2018-10-01 14:51:16</td>
                  
                  <td>14</td>
                  
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
                  
                  <td>3 Doors Down</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nicole</td>
                  
                  <td>F</td>
                  
                  <td>7</td>
                  
                  <td>Beck</td>
                  
                  <td>237.13914</td>
                  
                  <td>paid</td>
                  
                  <td>Vineland-Bridgeton, NJ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532224335000</td>
                  
                  <td>123</td>
                  
                  <td>Here Without You</td>
                  
                  <td>200</td>
                  
                  <td>1538408366000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                  <td>0</td>
                  
                  <td>NJ</td>
                  
                  <td>2018-10-01 15:39:26</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Emily</td>
                  
                  <td>F</td>
                  
                  <td>43</td>
                  
                  <td>Morrison</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1534367797000</td>
                  
                  <td>477</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538410672000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>232</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 16:17:52</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>114</td>
                  
                  <td>Myers</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Grand Rapids-Wyoming, MI</td>
                  
                  <td>PUT</td>
                  
                  <td>Add to Playlist</td>
                  
                  <td>1529995579000</td>
                  
                  <td>235</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538411368000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>236</td>
                  
                  <td>0</td>
                  
                  <td>MI</td>
                  
                  <td>2018-10-01 16:29:28</td>
                  
                  <td>16</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Golden Earring</td>
                  
                  <td>Logged In</td>
                  
                  <td>Adam</td>
                  
                  <td>M</td>
                  
                  <td>4</td>
                  
                  <td>Johnson</td>
                  
                  <td>302.65424</td>
                  
                  <td>free</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536986118000</td>
                  
                  <td>173</td>
                  
                  <td>Radar Love</td>
                  
                  <td>200</td>
                  
                  <td>1538411433000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>174</td>
                  
                  <td>0</td>
                  
                  <td>IL-IN-WI</td>
                  
                  <td>2018-10-01 16:30:33</td>
                  
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
                  
                  <td>M.I.A.</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>4</td>
                  
                  <td>Taylor</td>
                  
                  <td>206.13179</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533764798000</td>
                  
                  <td>562</td>
                  
                  <td>Paper Planes</td>
                  
                  <td>200</td>
                  
                  <td>1538412495000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
                  <td>0</td>
                  
                  <td>VA-NC</td>
                  
                  <td>2018-10-01 16:48:15</td>
                  
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
                  
                  <td>Millie Jackson</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>121</td>
                  
                  <td>Myers</td>
                  
                  <td>195.49995</td>
                  
                  <td>paid</td>
                  
                  <td>Grand Rapids-Wyoming, MI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529995579000</td>
                  
                  <td>235</td>
                  
                  <td>Watch The One Who Brings You The News</td>
                  
                  <td>200</td>
                  
                  <td>1538412601000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>236</td>
                  
                  <td>0</td>
                  
                  <td>MI</td>
                  
                  <td>2018-10-01 16:50:01</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Madvillain</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lakyla</td>
                  
                  <td>F</td>
                  
                  <td>49</td>
                  
                  <td>Porter</td>
                  
                  <td>182.77832</td>
                  
                  <td>paid</td>
                  
                  <td>Modesto, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535849930000</td>
                  
                  <td>507</td>
                  
                  <td>Money Folder</td>
                  
                  <td>200</td>
                  
                  <td>1538414043000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>162</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 17:14:03</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Blue Oyster Cult</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>101</td>
                  
                  <td>Campos</td>
                  
                  <td>388.20526</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000</td>
                  
                  <td>548</td>
                  
                  <td>Astronomy</td>
                  
                  <td>200</td>
                  
                  <td>1538416058000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                  <td>0</td>
                  
                  <td>AL</td>
                  
                  <td>2018-10-01 17:47:38</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Athlete</td>
                  
                  <td>Logged In</td>
                  
                  <td>Chase</td>
                  
                  <td>M</td>
                  
                  <td>4</td>
                  
                  <td>Ross</td>
                  
                  <td>260.25751</td>
                  
                  <td>free</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532450666000</td>
                  
                  <td>136</td>
                  
                  <td>Wires (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538416430000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.78.2 (KHTML, like Gecko) Version/7.0.6 Safari/537.78.2"</td>
                  
                  <td>137</td>
                  
                  <td>0</td>
                  
                  <td>NY-NJ-PA</td>
                  
                  <td>2018-10-01 17:53:50</td>
                  
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
                  
                  <td>Muse</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kinsley</td>
                  
                  <td>F</td>
                  
                  <td>3</td>
                  
                  <td>Joyce</td>
                  
                  <td>209.50159</td>
                  
                  <td>free</td>
                  
                  <td>Cedar City, UT</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536327321000</td>
                  
                  <td>527</td>
                  
                  <td>Supermassive Black Hole (Twilight Soundtrack Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538417147000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>241</td>
                  
                  <td>0</td>
                  
                  <td>UT</td>
                  
                  <td>2018-10-01 18:05:47</td>
                  
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
                  
                  <td>Julie London</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nicole</td>
                  
                  <td>F</td>
                  
                  <td>44</td>
                  
                  <td>Beck</td>
                  
                  <td>136.54159</td>
                  
                  <td>paid</td>
                  
                  <td>Vineland-Bridgeton, NJ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532224335000</td>
                  
                  <td>123</td>
                  
                  <td>You'd Be So Nice To Come Home To</td>
                  
                  <td>200</td>
                  
                  <td>1538417366000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                  <td>0</td>
                  
                  <td>NJ</td>
                  
                  <td>2018-10-01 18:09:26</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Jim Sturgess</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>156</td>
                  
                  <td>Myers</td>
                  
                  <td>137.06404</td>
                  
                  <td>paid</td>
                  
                  <td>Grand Rapids-Wyoming, MI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529995579000</td>
                  
                  <td>235</td>
                  
                  <td>Revolution</td>
                  
                  <td>200</td>
                  
                  <td>1538419392000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>236</td>
                  
                  <td>0</td>
                  
                  <td>MI</td>
                  
                  <td>2018-10-01 18:43:12</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>52</td>
                  
                  <td>Taylor</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1533764798000</td>
                  
                  <td>562</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538422096000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
                  <td>0</td>
                  
                  <td>VA-NC</td>
                  
                  <td>2018-10-01 19:28:16</td>
                  
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
                  
                  <td>Echo And The Bunnymen</td>
                  
                  <td>Logged In</td>
                  
                  <td>Madelyn</td>
                  
                  <td>F</td>
                  
                  <td>46</td>
                  
                  <td>Henson</td>
                  
                  <td>359.00036</td>
                  
                  <td>free</td>
                  
                  <td>Charlotte-Concord-Gastonia, NC-SC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532920994000</td>
                  
                  <td>112</td>
                  
                  <td>Over The Wall</td>
                  
                  <td>200</td>
                  
                  <td>1538422264000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>113</td>
                  
                  <td>0</td>
                  
                  <td>NC-SC</td>
                  
                  <td>2018-10-01 19:31:04</td>
                  
                  <td>19</td>
                  
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
                  
                  <td>The Crests</td>
                  
                  <td>Logged In</td>
                  
                  <td>Spencer</td>
                  
                  <td>M</td>
                  
                  <td>32</td>
                  
                  <td>Gonzalez</td>
                  
                  <td>182.88281</td>
                  
                  <td>free</td>
                  
                  <td>Concord, NH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537347211000</td>
                  
                  <td>64</td>
                  
                  <td>16 Candles</td>
                  
                  <td>200</td>
                  
                  <td>1538422685000</td>
                  
                  <td>Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>65</td>
                  
                  <td>0</td>
                  
                  <td>NH</td>
                  
                  <td>2018-10-01 19:38:05</td>
                  
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
                  
                  <td>Avril Lavigne</td>
                  
                  <td>Logged In</td>
                  
                  <td>Spencer</td>
                  
                  <td>M</td>
                  
                  <td>35</td>
                  
                  <td>Gonzalez</td>
                  
                  <td>242.41587</td>
                  
                  <td>free</td>
                  
                  <td>Concord, NH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537347211000</td>
                  
                  <td>64</td>
                  
                  <td>My Happy Ending</td>
                  
                  <td>200</td>
                  
                  <td>1538423068000</td>
                  
                  <td>Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>65</td>
                  
                  <td>0</td>
                  
                  <td>NH</td>
                  
                  <td>2018-10-01 19:44:28</td>
                  
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
                  
                  <td>Duran Duran</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lakyla</td>
                  
                  <td>F</td>
                  
                  <td>88</td>
                  
                  <td>Porter</td>
                  
                  <td>209.76281</td>
                  
                  <td>paid</td>
                  
                  <td>Modesto, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535849930000</td>
                  
                  <td>507</td>
                  
                  <td>Girls On Film</td>
                  
                  <td>200</td>
                  
                  <td>1538423135000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>162</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 19:45:35</td>
                  
                  <td>19</td>
                  
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
                  
                  <td>Radiohead</td>
                  
                  <td>Logged In</td>
                  
                  <td>Emily</td>
                  
                  <td>F</td>
                  
                  <td>110</td>
                  
                  <td>Morrison</td>
                  
                  <td>258.40281</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534367797000</td>
                  
                  <td>477</td>
                  
                  <td>Sail To The Moon</td>
                  
                  <td>200</td>
                  
                  <td>1538423189000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>232</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 19:46:29</td>
                  
                  <td>19</td>
                  
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
                  
                  <td>PanteÃÂ³n RococÃÂ³</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>177</td>
                  
                  <td>Myers</td>
                  
                  <td>207.04608</td>
                  
                  <td>paid</td>
                  
                  <td>Grand Rapids-Wyoming, MI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529995579000</td>
                  
                  <td>235</td>
                  
                  <td>Reality Shock</td>
                  
                  <td>200</td>
                  
                  <td>1538423713000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>236</td>
                  
                  <td>0</td>
                  
                  <td>MI</td>
                  
                  <td>2018-10-01 19:55:13</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Britney Spears</td>
                  
                  <td>Logged In</td>
                  
                  <td>Everett</td>
                  
                  <td>M</td>
                  
                  <td>23</td>
                  
                  <td>Quinn</td>
                  
                  <td>198.97424</td>
                  
                  <td>free</td>
                  
                  <td>Appleton, WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536082261000</td>
                  
                  <td>553</td>
                  
                  <td>Toxic</td>
                  
                  <td>200</td>
                  
                  <td>1538424198000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>195</td>
                  
                  <td>0</td>
                  
                  <td>WI</td>
                  
                  <td>2018-10-01 20:03:18</td>
                  
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
                  
                  <td>Todd Barry</td>
                  
                  <td>Logged In</td>
                  
                  <td>Emily</td>
                  
                  <td>F</td>
                  
                  <td>120</td>
                  
                  <td>Morrison</td>
                  
                  <td>126.82404</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534367797000</td>
                  
                  <td>477</td>
                  
                  <td>Sugar Ray (LP Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538425448000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>232</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 20:24:08</td>
                  
                  <td>20</td>
                  
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
                  
                  <td>Drake / Kanye West / Lil Wayne / Eminem</td>
                  
                  <td>Logged In</td>
                  
                  <td>Joseph</td>
                  
                  <td>M</td>
                  
                  <td>13</td>
                  
                  <td>Morales</td>
                  
                  <td>357.66812</td>
                  
                  <td>free</td>
                  
                  <td>Corpus Christi, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532063507000</td>
                  
                  <td>563</td>
                  
                  <td>Forever</td>
                  
                  <td>200</td>
                  
                  <td>1538425674000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>293</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 20:27:54</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Aurora</td>
                  
                  <td>F</td>
                  
                  <td>0</td>
                  
                  <td>Humphrey</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1536795126000</td>
                  
                  <td>537</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538426019000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>127</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 20:33:39</td>
                  
                  <td>20</td>
                  
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
                  
                  <td>Ra Ra Riot</td>
                  
                  <td>Logged In</td>
                  
                  <td>Everett</td>
                  
                  <td>M</td>
                  
                  <td>38</td>
                  
                  <td>Quinn</td>
                  
                  <td>152.71138</td>
                  
                  <td>free</td>
                  
                  <td>Appleton, WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536082261000</td>
                  
                  <td>553</td>
                  
                  <td>Can You Tell</td>
                  
                  <td>200</td>
                  
                  <td>1538427012000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>195</td>
                  
                  <td>0</td>
                  
                  <td>WI</td>
                  
                  <td>2018-10-01 20:50:12</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Emily</td>
                  
                  <td>F</td>
                  
                  <td>131</td>
                  
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
                  
                  <td>1538427524000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>232</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 20:58:44</td>
                  
                  <td>20</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Viviana</td>
                  
                  <td>F</td>
                  
                  <td>11</td>
                  
                  <td>Finley</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Gallup, NM</td>
                  
                  <td>GET</td>
                  
                  <td>Downgrade</td>
                  
                  <td>1523777521000</td>
                  
                  <td>454</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538427677000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>198</td>
                  
                  <td>0</td>
                  
                  <td>NM</td>
                  
                  <td>2018-10-01 21:01:17</td>
                  
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
                  
                  <td>Gipsy Kings</td>
                  
                  <td>Logged In</td>
                  
                  <td>Viviana</td>
                  
                  <td>F</td>
                  
                  <td>20</td>
                  
                  <td>Finley</td>
                  
                  <td>448.88771</td>
                  
                  <td>paid</td>
                  
                  <td>Gallup, NM</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1523777521000</td>
                  
                  <td>454</td>
                  
                  <td>Amor Gitano</td>
                  
                  <td>200</td>
                  
                  <td>1538430343000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>198</td>
                  
                  <td>0</td>
                  
                  <td>NM</td>
                  
                  <td>2018-10-01 21:45:43</td>
                  
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
                  
                  <td>Kings Of Leon</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nicole</td>
                  
                  <td>F</td>
                  
                  <td>119</td>
                  
                  <td>Beck</td>
                  
                  <td>201.79546</td>
                  
                  <td>paid</td>
                  
                  <td>Vineland-Bridgeton, NJ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532224335000</td>
                  
                  <td>123</td>
                  
                  <td>Revelry</td>
                  
                  <td>200</td>
                  
                  <td>1538432618000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                  <td>0</td>
                  
                  <td>NJ</td>
                  
                  <td>2018-10-01 22:23:38</td>
                  
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
                  
                  <td>R.E.M.</td>
                  
                  <td>Logged In</td>
                  
                  <td>Viviana</td>
                  
                  <td>F</td>
                  
                  <td>30</td>
                  
                  <td>Finley</td>
                  
                  <td>197.48526</td>
                  
                  <td>paid</td>
                  
                  <td>Gallup, NM</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1523777521000</td>
                  
                  <td>454</td>
                  
                  <td>The One I Love</td>
                  
                  <td>200</td>
                  
                  <td>1538432734000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>198</td>
                  
                  <td>0</td>
                  
                  <td>NM</td>
                  
                  <td>2018-10-01 22:25:34</td>
                  
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
                  
                  <td>Armchair Martian</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>218</td>
                  
                  <td>Myers</td>
                  
                  <td>113.21424</td>
                  
                  <td>paid</td>
                  
                  <td>Grand Rapids-Wyoming, MI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529995579000</td>
                  
                  <td>235</td>
                  
                  <td>Retardent (1997)</td>
                  
                  <td>200</td>
                  
                  <td>1538432974000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>236</td>
                  
                  <td>0</td>
                  
                  <td>MI</td>
                  
                  <td>2018-10-01 22:29:34</td>
                  
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
                  
                  <td>The Killers</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>32</td>
                  
                  <td>Hogan</td>
                  
                  <td>225.12281</td>
                  
                  <td>free</td>
                  
                  <td>Denver-Aurora-Lakewood, CO</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535066380000</td>
                  
                  <td>523</td>
                  
                  <td>A Dustland Fairytale</td>
                  
                  <td>200</td>
                  
                  <td>1538433311000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.2; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>101</td>
                  
                  <td>0</td>
                  
                  <td>CO</td>
                  
                  <td>2018-10-01 22:35:11</td>
                  
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
                  
                  <td>Brayden</td>
                  
                  <td>M</td>
                  
                  <td>45</td>
                  
                  <td>Thomas</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1534133898000</td>
                  
                  <td>556</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538434037000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>85</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 22:47:17</td>
                  
                  <td>22</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
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
                  
                  <td>83</td>
                  
                  <td>Quinn</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Appleton, WI</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1536082261000</td>
                  
                  <td>553</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538434620000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>195</td>
                  
                  <td>0</td>
                  
                  <td>WI</td>
                  
                  <td>2018-10-01 22:57:00</td>
                  
                  <td>22</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Brayden</td>
                  
                  <td>M</td>
                  
                  <td>53</td>
                  
                  <td>Thomas</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1534133898000</td>
                  
                  <td>556</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538436146000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>85</td>
                  
                  <td>0</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 23:22:26</td>
                  
                  <td>23</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Shins</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nicole</td>
                  
                  <td>F</td>
                  
                  <td>138</td>
                  
                  <td>Beck</td>
                  
                  <td>225.61914</td>
                  
                  <td>paid</td>
                  
                  <td>Vineland-Bridgeton, NJ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532224335000</td>
                  
                  <td>123</td>
                  
                  <td>Spilt Needles (Album)</td>
                  
                  <td>200</td>
                  
                  <td>1538437395000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                  <td>0</td>
                  
                  <td>NJ</td>
                  
                  <td>2018-10-01 23:43:15</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Aurora</td>
                  
                  <td>F</td>
                  
                  <td>51</td>
                  
                  <td>Humphrey</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>Add Friend</td>
                  
                  <td>1536795126000</td>
                  
                  <td>537</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538438127000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>127</td>
                  
                  <td>0</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-01 23:55:27</td>
                  
                  <td>23</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
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
      var tableWrapper = $('.df-table-wrapper-f0f0f373');
      var fixedHeader = $('.fixed-header', tableWrapper);
      var tableContainer = $('.df-table-container', tableWrapper);
      var table = $('.df-table', tableContainer);
      var rows = $('tbody > tr', table);
      var total = 100;
  
      fixedHeader
        .css('width', table.width())
        .find('.fixed-cell')
        .each(function(i, e) {
          $(this).css('width', $('.df-table-wrapper-f0f0f373 th:nth-child(' + (i+1) + ')').css('width'));
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



![png](output_17_1.png)


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


![png](output_19_0.png)


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



![png](output_23_1.png)


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
    <div id="chartFigure681284ee" class="pd_save is-viewer-good" style="overflow-x:auto">
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
  
  
  <div class="df-table-wrapper df-table-wrapper-681284ee panel-group pd_save">
    <!-- dataframe schema -->
    
    <div class="panel panel-default">
      <div class="panel-heading">
        <h4 class="panel-title" style="margin: 0px;">
          <a data-toggle="collapse" href="#df-schema-681284ee" data-parent="#df-table-wrapper-681284ee">Schema</a>
        </h4>
      </div>
      <div id="df-schema-681284ee" class="panel-collapse collapse">
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
          <a data-toggle="collapse" href="#df-table-681284ee" data-parent="#df-table-wrapper-681284ee"> Table</a>
        </h4>
      </div>
      
      <div id="df-table-681284ee" class="panel-collapse collapse in">
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
                  
                  <td>297</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>19</td>
                  
                  <td>45</td>
                  
                  <td>4447</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.013</td>
                  
                  <td>0.002</td>
                  
                  <td>0.0</td>
                  
                  <td>0.004</td>
                  
                  <td>0.002</td>
                  
                  <td>0.115</td>
                  
                  <td>0.002</td>
                  
                  <td>(12,[5,6,8,9,10,11],[0.013,0.002,0.004,0.002,0.115,0.002])</td>
                  
                  <td>[0.004308316738837159,-0.0024631478192377824,-0.012166427246806215]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100019</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>39</td>
                  
                  <td>3</td>
                  
                  <td>11</td>
                  
                  <td>8</td>
                  
                  <td>6</td>
                  
                  <td>201</td>
                  
                  <td>84</td>
                  
                  <td>49500</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.209</td>
                  
                  <td>0.037</td>
                  
                  <td>0.022</td>
                  
                  <td>0.051</td>
                  
                  <td>0.024</td>
                  
                  <td>0.024</td>
                  
                  <td>0.215</td>
                  
                  <td>0.024</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.209,0.037,0.022,0.051,0.024,0.024,0.215,0.024]</td>
                  
                  <td>[-0.07072607210530543,-1.0039241699521024,-0.055648359392813554]</td>
                  
                </tr>
                
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
                  
                  <td>[1.0385046549278183,-1.1109086413324312,0.03909680279864562]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>86</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>18</td>
                  
                  <td>18</td>
                  
                  <td>73</td>
                  
                  <td>34</td>
                  
                  <td>36</td>
                  
                  <td>1660</td>
                  
                  <td>137</td>
                  
                  <td>409073</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.096</td>
                  
                  <td>0.225</td>
                  
                  <td>0.144</td>
                  
                  <td>0.215</td>
                  
                  <td>0.145</td>
                  
                  <td>0.203</td>
                  
                  <td>0.351</td>
                  
                  <td>0.204</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.096,0.225,0.144,0.215,0.145,0.203,0.351,0.204]</td>
                  
                  <td>[1.0444080445722572,-1.115242564630351,0.013538838803837362]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100021</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>3</td>
                  
                  <td>34</td>
                  
                  <td>8</td>
                  
                  <td>28</td>
                  
                  <td>16</td>
                  
                  <td>14</td>
                  
                  <td>683</td>
                  
                  <td>64</td>
                  
                  <td>174690</td>
                  
                  <td>0.667</td>
                  
                  <td>1.0</td>
                  
                  <td>0.182</td>
                  
                  <td>0.1</td>
                  
                  <td>0.055</td>
                  
                  <td>0.101</td>
                  
                  <td>0.056</td>
                  
                  <td>0.083</td>
                  
                  <td>0.164</td>
                  
                  <td>0.087</td>
                  
                  <td>[1.0,1.0,0.667,1.0,0.182,0.1,0.055,0.101,0.056,0.083,0.164,0.087]</td>
                  
                  <td>[1.1407283640003463,-1.121606527570895,-0.2759761331325006]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>257</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>55</td>
                  
                  <td>54</td>
                  
                  <td>14972</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.027</td>
                  
                  <td>0.025</td>
                  
                  <td>0.004</td>
                  
                  <td>0.006</td>
                  
                  <td>0.004</td>
                  
                  <td>0.007</td>
                  
                  <td>0.138</td>
                  
                  <td>0.007</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.027,0.025,0.004,0.006,0.004,0.007,0.138,0.007]</td>
                  
                  <td>[0.009158259481997767,-0.004088641996182653,-0.028195916944504795]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300008</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>9</td>
                  
                  <td>5</td>
                  
                  <td>73</td>
                  
                  <td>17</td>
                  
                  <td>30</td>
                  
                  <td>793</td>
                  
                  <td>90</td>
                  
                  <td>195009</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.048</td>
                  
                  <td>0.062</td>
                  
                  <td>0.144</td>
                  
                  <td>0.108</td>
                  
                  <td>0.121</td>
                  
                  <td>0.097</td>
                  
                  <td>0.231</td>
                  
                  <td>0.097</td>
                  
                  <td>[0.0,1.0,0.333,0.333,0.048,0.062,0.144,0.108,0.121,0.097,0.231,0.097]</td>
                  
                  <td>[1.0802235417075026,-0.10103229857851309,0.017881740591075727]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>194</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>7</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>4</td>
                  
                  <td>3</td>
                  
                  <td>121</td>
                  
                  <td>77</td>
                  
                  <td>57617</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.037</td>
                  
                  <td>0.013</td>
                  
                  <td>0.01</td>
                  
                  <td>0.025</td>
                  
                  <td>0.012</td>
                  
                  <td>0.015</td>
                  
                  <td>0.197</td>
                  
                  <td>0.028</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.037,0.013,0.01,0.025,0.012,0.015,0.197,0.028]</td>
                  
                  <td>[-0.08212673186952205,-0.9996957984808627,0.005514176050036901]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>63</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>11</td>
                  
                  <td>40</td>
                  
                  <td>2256</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.006</td>
                  
                  <td>0.004</td>
                  
                  <td>0.001</td>
                  
                  <td>0.103</td>
                  
                  <td>0.001</td>
                  
                  <td>(12,[7,8,9,10,11],[0.006,0.004,0.001,0.103,0.001])</td>
                  
                  <td>[0.002310380404540388,-0.0017830300551152122,-0.007708556511142069]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>289</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>5</td>
                  
                  <td>37</td>
                  
                  <td>18</td>
                  
                  <td>22</td>
                  
                  <td>651</td>
                  
                  <td>150</td>
                  
                  <td>158491</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.043</td>
                  
                  <td>0.062</td>
                  
                  <td>0.073</td>
                  
                  <td>0.114</td>
                  
                  <td>0.089</td>
                  
                  <td>0.08</td>
                  
                  <td>0.385</td>
                  
                  <td>0.079</td>
                  
                  <td>[1.0,1.0,0.333,0.333,0.043,0.062,0.073,0.114,0.089,0.08,0.385,0.079]</td>
                  
                  <td>[0.9606441975398318,-1.092465182754234,0.09905912899328914]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300044</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>9</td>
                  
                  <td>82</td>
                  
                  <td>26</td>
                  
                  <td>18</td>
                  
                  <td>876</td>
                  
                  <td>125</td>
                  
                  <td>220433</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.005</td>
                  
                  <td>0.113</td>
                  
                  <td>0.162</td>
                  
                  <td>0.165</td>
                  
                  <td>0.073</td>
                  
                  <td>0.107</td>
                  
                  <td>0.321</td>
                  
                  <td>0.11</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.005,0.113,0.162,0.165,0.073,0.107,0.321,0.11]</td>
                  
                  <td>[0.8963916825296881,-1.083647712618816,0.2767515254796237]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>292</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>38</td>
                  
                  <td>138</td>
                  
                  <td>57</td>
                  
                  <td>88</td>
                  
                  <td>2954</td>
                  
                  <td>86</td>
                  
                  <td>734723</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.005</td>
                  
                  <td>0.475</td>
                  
                  <td>0.273</td>
                  
                  <td>0.361</td>
                  
                  <td>0.355</td>
                  
                  <td>0.361</td>
                  
                  <td>0.221</td>
                  
                  <td>0.367</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.005,0.475,0.273,0.361,0.355,0.361,0.221,0.367]</td>
                  
                  <td>[1.2167441489480069,-1.1502358873046539,-0.2703586984906697]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>48</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>6</td>
                  
                  <td>19</td>
                  
                  <td>8</td>
                  
                  <td>6</td>
                  
                  <td>302</td>
                  
                  <td>65</td>
                  
                  <td>79493</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.043</td>
                  
                  <td>0.075</td>
                  
                  <td>0.038</td>
                  
                  <td>0.051</td>
                  
                  <td>0.024</td>
                  
                  <td>0.037</td>
                  
                  <td>0.167</td>
                  
                  <td>0.039</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.043,0.075,0.038,0.051,0.024,0.037,0.167,0.039]</td>
                  
                  <td>[1.0016254055168432,-0.0890005173723516,0.2335646769638046]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>273</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>11</td>
                  
                  <td>14</td>
                  
                  <td>55</td>
                  
                  <td>28</td>
                  
                  <td>34</td>
                  
                  <td>1210</td>
                  
                  <td>108</td>
                  
                  <td>295011</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.059</td>
                  
                  <td>0.175</td>
                  
                  <td>0.109</td>
                  
                  <td>0.177</td>
                  
                  <td>0.137</td>
                  
                  <td>0.148</td>
                  
                  <td>0.277</td>
                  
                  <td>0.147</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.059,0.175,0.109,0.177,0.137,0.148,0.277,0.147]</td>
                  
                  <td>[1.004563535148611,-1.1055288715150633,0.0958349888688983]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>130</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>22</td>
                  
                  <td>2</td>
                  
                  <td>14</td>
                  
                  <td>8</td>
                  
                  <td>11</td>
                  
                  <td>272</td>
                  
                  <td>62</td>
                  
                  <td>69829</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.118</td>
                  
                  <td>0.025</td>
                  
                  <td>0.028</td>
                  
                  <td>0.051</td>
                  
                  <td>0.044</td>
                  
                  <td>0.033</td>
                  
                  <td>0.159</td>
                  
                  <td>0.035</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.118,0.025,0.028,0.051,0.044,0.033,0.159,0.035]</td>
                  
                  <td>[-0.06488736479013736,-1.0034971356599354,-0.04207328125456049]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>23</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>3</td>
                  
                  <td>1</td>
                  
                  <td>57</td>
                  
                  <td>119</td>
                  
                  <td>14079</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.016</td>
                  
                  <td>0.0</td>
                  
                  <td>0.006</td>
                  
                  <td>0.019</td>
                  
                  <td>0.004</td>
                  
                  <td>0.007</td>
                  
                  <td>0.305</td>
                  
                  <td>0.007</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.016,0.0,0.006,0.019,0.004,0.007,0.305,0.007]</td>
                  
                  <td>[0.007946076278861814,-0.005594594181712264,-0.028367818682902023]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>14</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>11</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>79</td>
                  
                  <td>124</td>
                  
                  <td>19221</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.059</td>
                  
                  <td>0.0</td>
                  
                  <td>0.006</td>
                  
                  <td>0.006</td>
                  
                  <td>0.004</td>
                  
                  <td>0.01</td>
                  
                  <td>0.318</td>
                  
                  <td>0.009</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.059,0.0,0.006,0.006,0.004,0.01,0.318,0.009]</td>
                  
                  <td>[-0.0929096221223217,-0.9990731281278438,0.01551760138734557]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200043</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>41</td>
                  
                  <td>22</td>
                  
                  <td>33</td>
                  
                  <td>12</td>
                  
                  <td>23</td>
                  
                  <td>615</td>
                  
                  <td>50</td>
                  
                  <td>173705</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.219</td>
                  
                  <td>0.275</td>
                  
                  <td>0.065</td>
                  
                  <td>0.076</td>
                  
                  <td>0.093</td>
                  
                  <td>0.075</td>
                  
                  <td>0.128</td>
                  
                  <td>0.086</td>
                  
                  <td>[0.0,0.0,0.0,0.333,0.219,0.275,0.065,0.076,0.093,0.075,0.128,0.086]</td>
                  
                  <td>[0.19085254820441958,-0.04403222236945004,-0.37790863067255714]</td>
                  
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
                  
                  <td>[-0.04285024995165698,-1.0106032331494776,-0.09393906674163113]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200050</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>3</td>
                  
                  <td>33</td>
                  
                  <td>35</td>
                  
                  <td>57</td>
                  
                  <td>33</td>
                  
                  <td>23</td>
                  
                  <td>1070</td>
                  
                  <td>46</td>
                  
                  <td>273123</td>
                  
                  <td>0.667</td>
                  
                  <td>1.0</td>
                  
                  <td>0.176</td>
                  
                  <td>0.438</td>
                  
                  <td>0.113</td>
                  
                  <td>0.209</td>
                  
                  <td>0.093</td>
                  
                  <td>0.131</td>
                  
                  <td>0.118</td>
                  
                  <td>0.136</td>
                  
                  <td>[1.0,1.0,0.667,1.0,0.176,0.438,0.113,0.209,0.093,0.131,0.118,0.136]</td>
                  
                  <td>[1.246617501791111,-1.1460491423243513,-0.47337938903046917]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>179</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>19</td>
                  
                  <td>94</td>
                  
                  <td>45</td>
                  
                  <td>59</td>
                  
                  <td>2219</td>
                  
                  <td>88</td>
                  
                  <td>546749</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.021</td>
                  
                  <td>0.237</td>
                  
                  <td>0.186</td>
                  
                  <td>0.285</td>
                  
                  <td>0.238</td>
                  
                  <td>0.271</td>
                  
                  <td>0.226</td>
                  
                  <td>0.273</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.021,0.237,0.186,0.285,0.238,0.271,0.226,0.273]</td>
                  
                  <td>[1.019916089403769,-1.109124714793985,0.056962939336554635]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200033</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>6</td>
                  
                  <td>38</td>
                  
                  <td>1199</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.025</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.001</td>
                  
                  <td>0.097</td>
                  
                  <td>0.0</td>
                  
                  <td>(12,[5,9,10],[0.025,0.001,0.097])</td>
                  
                  <td>[0.004868547529305651,-0.0024885258978708732,-0.01314278991476717]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>233</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>6</td>
                  
                  <td>16</td>
                  
                  <td>52</td>
                  
                  <td>11</td>
                  
                  <td>26</td>
                  
                  <td>993</td>
                  
                  <td>97</td>
                  
                  <td>255472</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.032</td>
                  
                  <td>0.2</td>
                  
                  <td>0.103</td>
                  
                  <td>0.07</td>
                  
                  <td>0.105</td>
                  
                  <td>0.121</td>
                  
                  <td>0.249</td>
                  
                  <td>0.127</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.032,0.2,0.103,0.07,0.105,0.121,0.249,0.127]</td>
                  
                  <td>[1.0779278858284742,-0.10652110345039112,0.09601734356538602]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200009</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>27</td>
                  
                  <td>11</td>
                  
                  <td>15</td>
                  
                  <td>8</td>
                  
                  <td>2</td>
                  
                  <td>336</td>
                  
                  <td>51</td>
                  
                  <td>83663</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.144</td>
                  
                  <td>0.138</td>
                  
                  <td>0.03</td>
                  
                  <td>0.051</td>
                  
                  <td>0.008</td>
                  
                  <td>0.041</td>
                  
                  <td>0.131</td>
                  
                  <td>0.041</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.144,0.138,0.03,0.051,0.008,0.041,0.131,0.041]</td>
                  
                  <td>[-0.04872638983507481,-1.0078226640212289,-0.08078479886478096]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100018</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>17</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>124</td>
                  
                  <td>60</td>
                  
                  <td>31034</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.091</td>
                  
                  <td>0.013</td>
                  
                  <td>0.016</td>
                  
                  <td>0.006</td>
                  
                  <td>0.004</td>
                  
                  <td>0.015</td>
                  
                  <td>0.154</td>
                  
                  <td>0.015</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.091,0.013,0.016,0.006,0.004,0.015,0.154,0.015]</td>
                  
                  <td>[-0.08821355224457168,-0.9983067918034242,0.005360435287513958]</td>
                  
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
                  
                  <td>[0.9076008313671179,-0.06976845704332275,0.3879202317950224]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>225</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>7</td>
                  
                  <td>31</td>
                  
                  <td>127</td>
                  
                  <td>67</td>
                  
                  <td>82</td>
                  
                  <td>2350</td>
                  
                  <td>72</td>
                  
                  <td>577855</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.037</td>
                  
                  <td>0.388</td>
                  
                  <td>0.251</td>
                  
                  <td>0.424</td>
                  
                  <td>0.331</td>
                  
                  <td>0.287</td>
                  
                  <td>0.185</td>
                  
                  <td>0.289</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.037,0.388,0.251,0.424,0.331,0.287,0.185,0.289]</td>
                  
                  <td>[1.1786888135111175,-1.1416130676516656,-0.2075162985825632]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>65</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>28</td>
                  
                  <td>15</td>
                  
                  <td>89</td>
                  
                  <td>36</td>
                  
                  <td>46</td>
                  
                  <td>1782</td>
                  
                  <td>71</td>
                  
                  <td>447333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.667</td>
                  
                  <td>0.15</td>
                  
                  <td>0.188</td>
                  
                  <td>0.176</td>
                  
                  <td>0.228</td>
                  
                  <td>0.185</td>
                  
                  <td>0.218</td>
                  
                  <td>0.182</td>
                  
                  <td>0.223</td>
                  
                  <td>[1.0,1.0,0.333,0.667,0.15,0.188,0.176,0.228,0.185,0.218,0.182,0.223]</td>
                  
                  <td>[1.1580093097387518,-1.1308733228467927,-0.24466317545809063]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>74</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>10</td>
                  
                  <td>20</td>
                  
                  <td>95</td>
                  
                  <td>33</td>
                  
                  <td>48</td>
                  
                  <td>1715</td>
                  
                  <td>55</td>
                  
                  <td>431367</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.053</td>
                  
                  <td>0.25</td>
                  
                  <td>0.188</td>
                  
                  <td>0.209</td>
                  
                  <td>0.194</td>
                  
                  <td>0.21</td>
                  
                  <td>0.141</td>
                  
                  <td>0.215</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.053,0.25,0.188,0.209,0.194,0.21,0.141,0.215]</td>
                  
                  <td>[1.1641290449515964,-0.1231741185470437,-0.05638787329078254]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200031</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>27</td>
                  
                  <td>164</td>
                  
                  <td>6319</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.016</td>
                  
                  <td>0.025</td>
                  
                  <td>0.004</td>
                  
                  <td>0.006</td>
                  
                  <td>0.004</td>
                  
                  <td>0.003</td>
                  
                  <td>0.421</td>
                  
                  <td>0.003</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.016,0.025,0.004,0.006,0.004,0.003,0.421,0.003]</td>
                  
                  <td>[-0.09030050486702092,-1.0008374831228302,0.01588535468117641]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>272</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>12</td>
                  
                  <td>4</td>
                  
                  <td>7</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>97</td>
                  
                  <td>29</td>
                  
                  <td>23179</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.064</td>
                  
                  <td>0.05</td>
                  
                  <td>0.014</td>
                  
                  <td>0.0</td>
                  
                  <td>0.012</td>
                  
                  <td>0.012</td>
                  
                  <td>0.074</td>
                  
                  <td>0.011</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.064,0.05,0.014,0.0,0.012,0.012,0.074,0.011]</td>
                  
                  <td>[0.016269121306700333,-0.00525189873830857,-0.04777063665410948]</td>
                  
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
                  
                  <td>[1.188829367339378,-1.132745893696252,-0.37407892031563916]</td>
                  
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
                  
                  <td>[-0.0644286669260728,-1.0044821874372447,-0.0518861158796017]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300050</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>6</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>61</td>
                  
                  <td>23</td>
                  
                  <td>14405</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.005</td>
                  
                  <td>0.0</td>
                  
                  <td>0.012</td>
                  
                  <td>0.0</td>
                  
                  <td>0.008</td>
                  
                  <td>0.007</td>
                  
                  <td>0.059</td>
                  
                  <td>0.007</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.005,0.0,0.012,0.0,0.008,0.007,0.059,0.007]</td>
                  
                  <td>[0.882342261677803,-0.06333700790856352,0.4359162839979893]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100043</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>187</td>
                  
                  <td>29</td>
                  
                  <td>80</td>
                  
                  <td>32</td>
                  
                  <td>44</td>
                  
                  <td>1568</td>
                  
                  <td>99</td>
                  
                  <td>441961</td>
                  
                  <td>0.667</td>
                  
                  <td>0.667</td>
                  
                  <td>1.0</td>
                  
                  <td>0.362</td>
                  
                  <td>0.158</td>
                  
                  <td>0.203</td>
                  
                  <td>0.177</td>
                  
                  <td>0.192</td>
                  
                  <td>0.254</td>
                  
                  <td>0.221</td>
                  
                  <td>[0.0,1.0,0.667,0.667,1.0,0.362,0.158,0.203,0.177,0.192,0.254,0.221]</td>
                  
                  <td>[1.2887730021101382,-0.1496656282898174,-0.6353307402646238]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300022</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>59</td>
                  
                  <td>10</td>
                  
                  <td>16</td>
                  
                  <td>691</td>
                  
                  <td>106</td>
                  
                  <td>169379</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.037</td>
                  
                  <td>0.117</td>
                  
                  <td>0.063</td>
                  
                  <td>0.065</td>
                  
                  <td>0.084</td>
                  
                  <td>0.272</td>
                  
                  <td>0.084</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.0,0.037,0.117,0.063,0.065,0.084,0.272,0.084]</td>
                  
                  <td>[0.8508656793473659,-1.0730050870666616,0.3626634206354148]</td>
                  
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
                  
                  <td>[1.0182094938359494,-0.0928856260620849,0.16580032404072248]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100049</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>37</td>
                  
                  <td>5</td>
                  
                  <td>13</td>
                  
                  <td>6</td>
                  
                  <td>22</td>
                  
                  <td>465</td>
                  
                  <td>51</td>
                  
                  <td>114115</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.198</td>
                  
                  <td>0.062</td>
                  
                  <td>0.026</td>
                  
                  <td>0.038</td>
                  
                  <td>0.089</td>
                  
                  <td>0.057</td>
                  
                  <td>0.131</td>
                  
                  <td>0.057</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.198,0.062,0.026,0.038,0.089,0.057,0.131,0.057]</td>
                  
                  <td>[0.9112270379987285,-1.0848982310954003,0.2362309222868705]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>40</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>27</td>
                  
                  <td>13</td>
                  
                  <td>78</td>
                  
                  <td>28</td>
                  
                  <td>38</td>
                  
                  <td>1313</td>
                  
                  <td>93</td>
                  
                  <td>323031</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.144</td>
                  
                  <td>0.163</td>
                  
                  <td>0.154</td>
                  
                  <td>0.177</td>
                  
                  <td>0.153</td>
                  
                  <td>0.16</td>
                  
                  <td>0.238</td>
                  
                  <td>0.161</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.144,0.163,0.154,0.177,0.153,0.16,0.238,0.161]</td>
                  
                  <td>[1.114797379679424,-0.11431198140732994,0.008576296895776263]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>164</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>31</td>
                  
                  <td>8</td>
                  
                  <td>64</td>
                  
                  <td>10</td>
                  
                  <td>35</td>
                  
                  <td>1270</td>
                  
                  <td>120</td>
                  
                  <td>319415</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.166</td>
                  
                  <td>0.1</td>
                  
                  <td>0.126</td>
                  
                  <td>0.063</td>
                  
                  <td>0.141</td>
                  
                  <td>0.155</td>
                  
                  <td>0.308</td>
                  
                  <td>0.159</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.166,0.1,0.126,0.063,0.141,0.155,0.308,0.159]</td>
                  
                  <td>[0.9795442361220228,-1.10064585338941,0.11860559263183157]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>161</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>4</td>
                  
                  <td>3</td>
                  
                  <td>13</td>
                  
                  <td>6</td>
                  
                  <td>10</td>
                  
                  <td>214</td>
                  
                  <td>57</td>
                  
                  <td>55232</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.021</td>
                  
                  <td>0.037</td>
                  
                  <td>0.026</td>
                  
                  <td>0.038</td>
                  
                  <td>0.04</td>
                  
                  <td>0.026</td>
                  
                  <td>0.146</td>
                  
                  <td>0.027</td>
                  
                  <td>[1.0,0.0,0.0,0.333,0.021,0.037,0.026,0.038,0.04,0.026,0.146,0.027]</td>
                  
                  <td>[0.013893954413244182,-1.0176567033531883,-0.13577431645347524]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>160</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>57</td>
                  
                  <td>9</td>
                  
                  <td>47</td>
                  
                  <td>16</td>
                  
                  <td>17</td>
                  
                  <td>1063</td>
                  
                  <td>79</td>
                  
                  <td>265310</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.305</td>
                  
                  <td>0.113</td>
                  
                  <td>0.093</td>
                  
                  <td>0.101</td>
                  
                  <td>0.069</td>
                  
                  <td>0.13</td>
                  
                  <td>0.203</td>
                  
                  <td>0.132</td>
                  
                  <td>[1.0,0.0,0.333,0.333,0.305,0.113,0.093,0.101,0.069,0.13,0.203,0.132]</td>
                  
                  <td>[0.10493545328709689,-1.0355082228568122,-0.44171245703593265]</td>
                  
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
                  
                  <td>[0.12126483300399402,-1.039420428784157,-0.4403058019470438]</td>
                  
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
                  
                  <td>[0.017153488683335133,-0.005800949190548692,-0.057534839164784635]</td>
                  
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
                  
                  <td>[0.9923644830544319,-1.0995207155201208,0.13486716211103583]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>167</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>16</td>
                  
                  <td>4</td>
                  
                  <td>10</td>
                  
                  <td>0</td>
                  
                  <td>8</td>
                  
                  <td>199</td>
                  
                  <td>137</td>
                  
                  <td>46289</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.086</td>
                  
                  <td>0.05</td>
                  
                  <td>0.02</td>
                  
                  <td>0.0</td>
                  
                  <td>0.032</td>
                  
                  <td>0.024</td>
                  
                  <td>0.351</td>
                  
                  <td>0.023</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.086,0.05,0.02,0.0,0.032,0.024,0.351,0.023]</td>
                  
                  <td>[0.02526242973700666,-0.010742406911182383,-0.07806254432492565]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>236</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>20</td>
                  
                  <td>19</td>
                  
                  <td>10</td>
                  
                  <td>466</td>
                  
                  <td>99</td>
                  
                  <td>119805</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.037</td>
                  
                  <td>0.04</td>
                  
                  <td>0.12</td>
                  
                  <td>0.04</td>
                  
                  <td>0.057</td>
                  
                  <td>0.254</td>
                  
                  <td>0.06</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.0,0.037,0.04,0.12,0.04,0.057,0.254,0.06]</td>
                  
                  <td>[0.8353708838251523,-1.0701051561858808,0.38857283341316917]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>165</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>11</td>
                  
                  <td>3</td>
                  
                  <td>19</td>
                  
                  <td>5</td>
                  
                  <td>8</td>
                  
                  <td>285</td>
                  
                  <td>89</td>
                  
                  <td>70350</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.059</td>
                  
                  <td>0.037</td>
                  
                  <td>0.038</td>
                  
                  <td>0.032</td>
                  
                  <td>0.032</td>
                  
                  <td>0.035</td>
                  
                  <td>0.228</td>
                  
                  <td>0.035</td>
                  
                  <td>[1.0,0.0,0.333,0.333,0.059,0.037,0.038,0.032,0.032,0.035,0.228,0.035]</td>
                  
                  <td>[0.03693981385001258,-1.0193654275277662,-0.26514065589488567]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200018</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>41</td>
                  
                  <td>34</td>
                  
                  <td>47</td>
                  
                  <td>22</td>
                  
                  <td>37</td>
                  
                  <td>1044</td>
                  
                  <td>77</td>
                  
                  <td>251765</td>
                  
                  <td>0.333</td>
                  
                  <td>0.667</td>
                  
                  <td>0.219</td>
                  
                  <td>0.425</td>
                  
                  <td>0.093</td>
                  
                  <td>0.139</td>
                  
                  <td>0.149</td>
                  
                  <td>0.128</td>
                  
                  <td>0.197</td>
                  
                  <td>0.125</td>
                  
                  <td>[1.0,0.0,0.333,0.667,0.219,0.425,0.093,0.139,0.149,0.128,0.197,0.125]</td>
                  
                  <td>[0.258042381717693,-1.0683994695593741,-0.6820999375536416]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>122</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>17</td>
                  
                  <td>3</td>
                  
                  <td>13</td>
                  
                  <td>6</td>
                  
                  <td>4</td>
                  
                  <td>286</td>
                  
                  <td>86</td>
                  
                  <td>70958</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.091</td>
                  
                  <td>0.037</td>
                  
                  <td>0.026</td>
                  
                  <td>0.038</td>
                  
                  <td>0.016</td>
                  
                  <td>0.035</td>
                  
                  <td>0.221</td>
                  
                  <td>0.035</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.091,0.037,0.026,0.038,0.016,0.035,0.221,0.035]</td>
                  
                  <td>[0.030624813441230213,-0.010081508508707215,-0.08350271673439491]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>129</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
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
                  
                  <td>[1.0,1.0,0.667,0.667,0.289,0.237,0.196,0.146,0.238,0.237,0.156,0.238]</td>
                  
                  <td>[1.1900789451922515,-1.1343308643944878,-0.4106229866933145]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>88</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>12</td>
                  
                  <td>21</td>
                  
                  <td>104</td>
                  
                  <td>49</td>
                  
                  <td>51</td>
                  
                  <td>1975</td>
                  
                  <td>56</td>
                  
                  <td>491844</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.064</td>
                  
                  <td>0.263</td>
                  
                  <td>0.206</td>
                  
                  <td>0.31</td>
                  
                  <td>0.206</td>
                  
                  <td>0.241</td>
                  
                  <td>0.144</td>
                  
                  <td>0.246</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.064,0.263,0.206,0.31,0.206,0.241,0.144,0.246]</td>
                  
                  <td>[1.1970072521923953,-0.13016245775863844,-0.11717372144875643]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>231</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>202</td>
                  
                  <td>58</td>
                  
                  <td>53280</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.005</td>
                  
                  <td>0.013</td>
                  
                  <td>0.016</td>
                  
                  <td>0.006</td>
                  
                  <td>0.008</td>
                  
                  <td>0.025</td>
                  
                  <td>0.149</td>
                  
                  <td>0.026</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.005,0.013,0.016,0.006,0.008,0.025,0.149,0.026]</td>
                  
                  <td>[0.8926596199153339,-0.06668215130795802,0.41428294061921456]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>162</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>3</td>
                  
                  <td>15</td>
                  
                  <td>95</td>
                  
                  <td>30</td>
                  
                  <td>41</td>
                  
                  <td>1676</td>
                  
                  <td>71</td>
                  
                  <td>425673</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.016</td>
                  
                  <td>0.188</td>
                  
                  <td>0.188</td>
                  
                  <td>0.19</td>
                  
                  <td>0.165</td>
                  
                  <td>0.205</td>
                  
                  <td>0.182</td>
                  
                  <td>0.212</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.016,0.188,0.188,0.19,0.165,0.205,0.182,0.212]</td>
                  
                  <td>[1.145074663419287,-0.11877380519485495,-0.01317715931787132]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>198</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>10</td>
                  
                  <td>69</td>
                  
                  <td>13</td>
                  
                  <td>30</td>
                  
                  <td>1248</td>
                  
                  <td>193</td>
                  
                  <td>310883</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.005</td>
                  
                  <td>0.125</td>
                  
                  <td>0.136</td>
                  
                  <td>0.082</td>
                  
                  <td>0.121</td>
                  
                  <td>0.153</td>
                  
                  <td>0.495</td>
                  
                  <td>0.155</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.005,0.125,0.136,0.082,0.121,0.153,0.495,0.155]</td>
                  
                  <td>[1.0042313116040034,-0.09444695563201189,0.20475365120348343]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>37</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>50</td>
                  
                  <td>1</td>
                  
                  <td>37</td>
                  
                  <td>12</td>
                  
                  <td>15</td>
                  
                  <td>492</td>
                  
                  <td>96</td>
                  
                  <td>120368</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.267</td>
                  
                  <td>0.013</td>
                  
                  <td>0.073</td>
                  
                  <td>0.076</td>
                  
                  <td>0.06</td>
                  
                  <td>0.06</td>
                  
                  <td>0.246</td>
                  
                  <td>0.06</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.267,0.013,0.073,0.076,0.06,0.06,0.246,0.06]</td>
                  
                  <td>[-0.04581514621089213,-1.0093757581256457,-0.11138558151657073]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>131</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>40</td>
                  
                  <td>9</td>
                  
                  <td>50</td>
                  
                  <td>21</td>
                  
                  <td>23</td>
                  
                  <td>949</td>
                  
                  <td>103</td>
                  
                  <td>234600</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.214</td>
                  
                  <td>0.113</td>
                  
                  <td>0.099</td>
                  
                  <td>0.133</td>
                  
                  <td>0.093</td>
                  
                  <td>0.116</td>
                  
                  <td>0.264</td>
                  
                  <td>0.117</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.214,0.113,0.099,0.133,0.093,0.116,0.264,0.117]</td>
                  
                  <td>[0.9669470406346983,-1.09829629987593,0.1289518265837372]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>36</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>10</td>
                  
                  <td>21</td>
                  
                  <td>103</td>
                  
                  <td>43</td>
                  
                  <td>69</td>
                  
                  <td>2099</td>
                  
                  <td>112</td>
                  
                  <td>517194</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.053</td>
                  
                  <td>0.263</td>
                  
                  <td>0.204</td>
                  
                  <td>0.272</td>
                  
                  <td>0.278</td>
                  
                  <td>0.257</td>
                  
                  <td>0.287</td>
                  
                  <td>0.258</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.053,0.263,0.204,0.272,0.278,0.257,0.287,0.258]</td>
                  
                  <td>[1.2073729666169883,-0.13401304762386423,-0.13701107181426586]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200036</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>21</td>
                  
                  <td>9</td>
                  
                  <td>14</td>
                  
                  <td>3</td>
                  
                  <td>8</td>
                  
                  <td>270</td>
                  
                  <td>98</td>
                  
                  <td>63362</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.112</td>
                  
                  <td>0.113</td>
                  
                  <td>0.028</td>
                  
                  <td>0.019</td>
                  
                  <td>0.032</td>
                  
                  <td>0.033</td>
                  
                  <td>0.251</td>
                  
                  <td>0.031</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.112,0.113,0.028,0.019,0.032,0.033,0.251,0.031]</td>
                  
                  <td>[-0.056826301781440554,-1.0071549811286284,-0.06210973354878846]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200008</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>8</td>
                  
                  <td>15</td>
                  
                  <td>4</td>
                  
                  <td>3</td>
                  
                  <td>224</td>
                  
                  <td>91</td>
                  
                  <td>54252</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.027</td>
                  
                  <td>0.1</td>
                  
                  <td>0.03</td>
                  
                  <td>0.025</td>
                  
                  <td>0.012</td>
                  
                  <td>0.027</td>
                  
                  <td>0.233</td>
                  
                  <td>0.027</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.027,0.1,0.03,0.025,0.012,0.027,0.233,0.027]</td>
                  
                  <td>[0.9955036633907663,-0.08873251334897987,0.24429068222330613]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>264</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>12</td>
                  
                  <td>7</td>
                  
                  <td>52</td>
                  
                  <td>18</td>
                  
                  <td>25</td>
                  
                  <td>920</td>
                  
                  <td>73</td>
                  
                  <td>222530</td>
                  
                  <td>0.667</td>
                  
                  <td>0.667</td>
                  
                  <td>0.064</td>
                  
                  <td>0.087</td>
                  
                  <td>0.103</td>
                  
                  <td>0.114</td>
                  
                  <td>0.101</td>
                  
                  <td>0.112</td>
                  
                  <td>0.187</td>
                  
                  <td>0.111</td>
                  
                  <td>[0.0,0.0,0.667,0.667,0.064,0.087,0.103,0.114,0.101,0.112,0.187,0.111]</td>
                  
                  <td>[0.3060724653198931,-0.05518444736641834,-0.669903730037797]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>216</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>15</td>
                  
                  <td>80</td>
                  
                  <td>37</td>
                  
                  <td>52</td>
                  
                  <td>1695</td>
                  
                  <td>116</td>
                  
                  <td>428958</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.043</td>
                  
                  <td>0.188</td>
                  
                  <td>0.158</td>
                  
                  <td>0.234</td>
                  
                  <td>0.21</td>
                  
                  <td>0.207</td>
                  
                  <td>0.297</td>
                  
                  <td>0.214</td>
                  
                  <td>[0.0,1.0,0.333,0.333,0.043,0.188,0.158,0.234,0.21,0.207,0.297,0.214]</td>
                  
                  <td>[1.175777791123224,-0.12272358668773443,-0.15521096882523927]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>68</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>0</td>
                  
                  <td>6</td>
                  
                  <td>2</td>
                  
                  <td>1</td>
                  
                  <td>52</td>
                  
                  <td>90</td>
                  
                  <td>13307</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.011</td>
                  
                  <td>0.0</td>
                  
                  <td>0.012</td>
                  
                  <td>0.013</td>
                  
                  <td>0.004</td>
                  
                  <td>0.006</td>
                  
                  <td>0.231</td>
                  
                  <td>0.006</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.011,0.0,0.012,0.013,0.004,0.006,0.231,0.006]</td>
                  
                  <td>[0.007348036774595331,-0.00445793988175583,-0.02343539280651241]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200014</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>4</td>
                  
                  <td>3</td>
                  
                  <td>3</td>
                  
                  <td>3</td>
                  
                  <td>3</td>
                  
                  <td>101</td>
                  
                  <td>101</td>
                  
                  <td>25036</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.021</td>
                  
                  <td>0.037</td>
                  
                  <td>0.006</td>
                  
                  <td>0.019</td>
                  
                  <td>0.012</td>
                  
                  <td>0.012</td>
                  
                  <td>0.259</td>
                  
                  <td>0.012</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.021,0.037,0.006,0.019,0.012,0.012,0.259,0.012]</td>
                  
                  <td>[0.875792739168476,-1.0775972008848578,0.3343733103033843]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>282</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>37</td>
                  
                  <td>29</td>
                  
                  <td>133</td>
                  
                  <td>79</td>
                  
                  <td>66</td>
                  
                  <td>2629</td>
                  
                  <td>78</td>
                  
                  <td>664850</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.198</td>
                  
                  <td>0.362</td>
                  
                  <td>0.263</td>
                  
                  <td>0.5</td>
                  
                  <td>0.266</td>
                  
                  <td>0.321</td>
                  
                  <td>0.2</td>
                  
                  <td>0.332</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.198,0.362,0.263,0.5,0.266,0.321,0.2,0.332]</td>
                  
                  <td>[1.1892942694699102,-1.1449398506352682,-0.26268090780697173]</td>
                  
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
                  
                  <td>[1.3328771210234889,-0.15561607272981554,-0.4417205104461437]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>38</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>35</td>
                  
                  <td>129</td>
                  
                  <td>44</td>
                  
                  <td>86</td>
                  
                  <td>2700</td>
                  
                  <td>75</td>
                  
                  <td>671166</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.438</td>
                  
                  <td>0.255</td>
                  
                  <td>0.278</td>
                  
                  <td>0.347</td>
                  
                  <td>0.33</td>
                  
                  <td>0.192</td>
                  
                  <td>0.335</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.0,0.438,0.255,0.278,0.347,0.33,0.192,0.335]</td>
                  
                  <td>[1.1014046948645488,-1.1269851503115755,-0.0867274599356174]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>11</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
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
                  
                  <td>[0.0,0.0,0.0,0.333,0.053,0.013,0.026,0.038,0.02,0.024,0.313,0.023]</td>
                  
                  <td>[0.10539593981373828,-0.024870475297881906,-0.1858369167518028]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>67</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>18</td>
                  
                  <td>57</td>
                  
                  <td>22</td>
                  
                  <td>45</td>
                  
                  <td>1178</td>
                  
                  <td>149</td>
                  
                  <td>289170</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.043</td>
                  
                  <td>0.225</td>
                  
                  <td>0.113</td>
                  
                  <td>0.139</td>
                  
                  <td>0.181</td>
                  
                  <td>0.144</td>
                  
                  <td>0.382</td>
                  
                  <td>0.144</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.043,0.225,0.113,0.139,0.181,0.144,0.382,0.144]</td>
                  
                  <td>[1.0141685887270098,-1.10920287201135,0.0771573106810553]</td>
                  
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
                  
                  <td>[1.0505771191375448,-1.1140355939797333,-0.003509550481113309]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>209</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>12</td>
                  
                  <td>59</td>
                  
                  <td>20</td>
                  
                  <td>48</td>
                  
                  <td>1499</td>
                  
                  <td>61</td>
                  
                  <td>388847</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.0</td>
                  
                  <td>0.15</td>
                  
                  <td>0.117</td>
                  
                  <td>0.127</td>
                  
                  <td>0.194</td>
                  
                  <td>0.183</td>
                  
                  <td>0.156</td>
                  
                  <td>0.194</td>
                  
                  <td>[0.0,1.0,0.333,0.333,0.0,0.15,0.117,0.127,0.194,0.183,0.156,0.194]</td>
                  
                  <td>[1.1361756597308135,-0.11218124142270594,-0.06938990213361111]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>147</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>31</td>
                  
                  <td>2</td>
                  
                  <td>12</td>
                  
                  <td>4</td>
                  
                  <td>8</td>
                  
                  <td>272</td>
                  
                  <td>68</td>
                  
                  <td>63554</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.166</td>
                  
                  <td>0.025</td>
                  
                  <td>0.024</td>
                  
                  <td>0.025</td>
                  
                  <td>0.032</td>
                  
                  <td>0.033</td>
                  
                  <td>0.174</td>
                  
                  <td>0.031</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.166,0.025,0.024,0.025,0.032,0.033,0.174,0.031]</td>
                  
                  <td>[-0.0724401970391913,-1.0025230975648298,-0.04039479627726978]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>280</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>7</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>0</td>
                  
                  <td>49</td>
                  
                  <td>100</td>
                  
                  <td>10699</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.037</td>
                  
                  <td>0.0</td>
                  
                  <td>0.002</td>
                  
                  <td>0.032</td>
                  
                  <td>0.0</td>
                  
                  <td>0.006</td>
                  
                  <td>0.256</td>
                  
                  <td>0.005</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.037,0.0,0.002,0.032,0.0,0.006,0.256,0.005]</td>
                  
                  <td>[-0.09150592255131924,-0.9984495191484699,0.020018415715414923]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>286</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>3</td>
                  
                  <td>2</td>
                  
                  <td>7</td>
                  
                  <td>0</td>
                  
                  <td>7</td>
                  
                  <td>245</td>
                  
                  <td>63</td>
                  
                  <td>60842</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.016</td>
                  
                  <td>0.025</td>
                  
                  <td>0.014</td>
                  
                  <td>0.0</td>
                  
                  <td>0.028</td>
                  
                  <td>0.03</td>
                  
                  <td>0.162</td>
                  
                  <td>0.03</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.016,0.025,0.014,0.0,0.028,0.03,0.162,0.03]</td>
                  
                  <td>[0.9795059972934397,-0.0837342426146224,0.2809796343514072]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>285</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>9</td>
                  
                  <td>5</td>
                  
                  <td>34</td>
                  
                  <td>19</td>
                  
                  <td>21</td>
                  
                  <td>563</td>
                  
                  <td>113</td>
                  
                  <td>134703</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.048</td>
                  
                  <td>0.062</td>
                  
                  <td>0.067</td>
                  
                  <td>0.12</td>
                  
                  <td>0.085</td>
                  
                  <td>0.069</td>
                  
                  <td>0.29</td>
                  
                  <td>0.067</td>
                  
                  <td>[0.0,0.0,0.333,0.333,0.048,0.062,0.067,0.12,0.085,0.069,0.29,0.067]</td>
                  
                  <td>[0.17842612470514246,-0.035617421785513886,-0.39069687428399336]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>113</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>24</td>
                  
                  <td>18</td>
                  
                  <td>44</td>
                  
                  <td>8</td>
                  
                  <td>32</td>
                  
                  <td>976</td>
                  
                  <td>121</td>
                  
                  <td>240879</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.128</td>
                  
                  <td>0.225</td>
                  
                  <td>0.087</td>
                  
                  <td>0.051</td>
                  
                  <td>0.129</td>
                  
                  <td>0.119</td>
                  
                  <td>0.31</td>
                  
                  <td>0.12</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.128,0.225,0.087,0.051,0.129,0.119,0.31,0.12]</td>
                  
                  <td>[1.0785196963547357,-0.10857487125314087,0.06959523844320534]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>177</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>19</td>
                  
                  <td>48</td>
                  
                  <td>4324</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.021</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.002</td>
                  
                  <td>0.123</td>
                  
                  <td>0.002</td>
                  
                  <td>(12,[0,4,9,10,11],[1.0,0.021,0.002,0.123,0.002])</td>
                  
                  <td>[-0.09837046830895006,-0.9952090239982593,0.040585938236297976]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200046</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>10</td>
                  
                  <td>27</td>
                  
                  <td>11</td>
                  
                  <td>20</td>
                  
                  <td>454</td>
                  
                  <td>65</td>
                  
                  <td>122374</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.043</td>
                  
                  <td>0.125</td>
                  
                  <td>0.053</td>
                  
                  <td>0.07</td>
                  
                  <td>0.081</td>
                  
                  <td>0.055</td>
                  
                  <td>0.167</td>
                  
                  <td>0.061</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.043,0.125,0.053,0.07,0.081,0.055,0.167,0.061]</td>
                  
                  <td>[1.031117921807113,-0.09553874942193596,0.18026441912618751]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200002</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>11</td>
                  
                  <td>5</td>
                  
                  <td>15</td>
                  
                  <td>2</td>
                  
                  <td>6</td>
                  
                  <td>310</td>
                  
                  <td>54</td>
                  
                  <td>78918</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.059</td>
                  
                  <td>0.062</td>
                  
                  <td>0.03</td>
                  
                  <td>0.013</td>
                  
                  <td>0.024</td>
                  
                  <td>0.038</td>
                  
                  <td>0.138</td>
                  
                  <td>0.039</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.059,0.062,0.03,0.013,0.024,0.038,0.138,0.039]</td>
                  
                  <td>[0.8927808689943446,-1.0799655536116477,0.29978355045069965]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>102</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>7</td>
                  
                  <td>1</td>
                  
                  <td>7</td>
                  
                  <td>5</td>
                  
                  <td>4</td>
                  
                  <td>124</td>
                  
                  <td>52</td>
                  
                  <td>29956</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.037</td>
                  
                  <td>0.013</td>
                  
                  <td>0.014</td>
                  
                  <td>0.032</td>
                  
                  <td>0.016</td>
                  
                  <td>0.015</td>
                  
                  <td>0.133</td>
                  
                  <td>0.015</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.037,0.013,0.014,0.032,0.016,0.015,0.133,0.015]</td>
                  
                  <td>[-0.0822365862254136,-0.9988691044070203,0.007929023835920469]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300001</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>3</td>
                  
                  <td>9</td>
                  
                  <td>11</td>
                  
                  <td>140</td>
                  
                  <td>29</td>
                  
                  <td>39</td>
                  
                  <td>1623</td>
                  
                  <td>195</td>
                  
                  <td>399589</td>
                  
                  <td>0.667</td>
                  
                  <td>1.0</td>
                  
                  <td>0.048</td>
                  
                  <td>0.138</td>
                  
                  <td>0.277</td>
                  
                  <td>0.184</td>
                  
                  <td>0.157</td>
                  
                  <td>0.198</td>
                  
                  <td>0.5</td>
                  
                  <td>0.199</td>
                  
                  <td>[0.0,1.0,0.667,1.0,0.048,0.138,0.277,0.184,0.157,0.198,0.5,0.199]</td>
                  
                  <td>[1.3499385537080792,-0.1531701932458789,-0.4983418267269785]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>73</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>20</td>
                  
                  <td>1</td>
                  
                  <td>9</td>
                  
                  <td>4</td>
                  
                  <td>5</td>
                  
                  <td>240</td>
                  
                  <td>87</td>
                  
                  <td>57871</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.107</td>
                  
                  <td>0.013</td>
                  
                  <td>0.018</td>
                  
                  <td>0.025</td>
                  
                  <td>0.02</td>
                  
                  <td>0.029</td>
                  
                  <td>0.223</td>
                  
                  <td>0.029</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.107,0.013,0.018,0.025,0.02,0.029,0.223,0.029]</td>
                  
                  <td>[0.021730220087038954,-0.008202862842771743,-0.07061413768285102]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>228</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>40</td>
                  
                  <td>34</td>
                  
                  <td>154</td>
                  
                  <td>88</td>
                  
                  <td>99</td>
                  
                  <td>3389</td>
                  
                  <td>115</td>
                  
                  <td>827138</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.214</td>
                  
                  <td>0.425</td>
                  
                  <td>0.304</td>
                  
                  <td>0.557</td>
                  
                  <td>0.399</td>
                  
                  <td>0.414</td>
                  
                  <td>0.295</td>
                  
                  <td>0.413</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.214,0.425,0.304,0.557,0.399,0.414,0.295,0.413]</td>
                  
                  <td>[1.2653915221408545,-1.1623726261216016,-0.40383751533812196]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>3</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>24</td>
                  
                  <td>110</td>
                  
                  <td>6254</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.005</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.003</td>
                  
                  <td>0.282</td>
                  
                  <td>0.003</td>
                  
                  <td>(12,[0,4,9,10,11],[1.0,0.005,0.003,0.282,0.003])</td>
                  
                  <td>[-0.0972641534023592,-0.9973082675373606,0.03668490672575561]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>212</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>11</td>
                  
                  <td>46</td>
                  
                  <td>12</td>
                  
                  <td>21</td>
                  
                  <td>1013</td>
                  
                  <td>90</td>
                  
                  <td>249047</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.005</td>
                  
                  <td>0.138</td>
                  
                  <td>0.091</td>
                  
                  <td>0.076</td>
                  
                  <td>0.085</td>
                  
                  <td>0.124</td>
                  
                  <td>0.231</td>
                  
                  <td>0.124</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.005,0.138,0.091,0.076,0.085,0.124,0.231,0.124]</td>
                  
                  <td>[0.981888193988427,-0.08692642338988933,0.25149478619254434]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300009</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>6</td>
                  
                  <td>90</td>
                  
                  <td>20</td>
                  
                  <td>19</td>
                  
                  <td>854</td>
                  
                  <td>98</td>
                  
                  <td>212238</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.011</td>
                  
                  <td>0.075</td>
                  
                  <td>0.178</td>
                  
                  <td>0.127</td>
                  
                  <td>0.077</td>
                  
                  <td>0.104</td>
                  
                  <td>0.251</td>
                  
                  <td>0.106</td>
                  
                  <td>[0.0,1.0,0.0,0.333,0.011,0.075,0.178,0.127,0.077,0.104,0.251,0.106]</td>
                  
                  <td>[1.0663115849809155,-0.10231983195125698,0.12716716389530716]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>57</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>7</td>
                  
                  <td>3</td>
                  
                  <td>6</td>
                  
                  <td>176</td>
                  
                  <td>44</td>
                  
                  <td>43629</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.025</td>
                  
                  <td>0.014</td>
                  
                  <td>0.019</td>
                  
                  <td>0.024</td>
                  
                  <td>0.021</td>
                  
                  <td>0.113</td>
                  
                  <td>0.021</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.0,0.025,0.014,0.019,0.024,0.021,0.113,0.021]</td>
                  
                  <td>[0.7979853852134496,-1.0606041198513048,0.4593703336719667]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>275</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>32</td>
                  
                  <td>7</td>
                  
                  <td>26</td>
                  
                  <td>14</td>
                  
                  <td>22</td>
                  
                  <td>647</td>
                  
                  <td>84</td>
                  
                  <td>164101</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.171</td>
                  
                  <td>0.087</td>
                  
                  <td>0.051</td>
                  
                  <td>0.089</td>
                  
                  <td>0.089</td>
                  
                  <td>0.079</td>
                  
                  <td>0.215</td>
                  
                  <td>0.082</td>
                  
                  <td>[1.0,0.0,0.0,0.333,0.171,0.087,0.051,0.089,0.089,0.079,0.215,0.082]</td>
                  
                  <td>[0.05902929784628261,-1.0294440627769121,-0.25309554039829707]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>10</td>
                  
                  <td>6</td>
                  
                  <td>5</td>
                  
                  <td>164</td>
                  
                  <td>7</td>
                  
                  <td>38062</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.025</td>
                  
                  <td>0.02</td>
                  
                  <td>0.038</td>
                  
                  <td>0.02</td>
                  
                  <td>0.02</td>
                  
                  <td>0.018</td>
                  
                  <td>0.019</td>
                  
                  <td>[1.0,1.0,0.0,0.0,0.0,0.025,0.02,0.038,0.02,0.02,0.018,0.019]</td>
                  
                  <td>[0.8003618249864759,-1.0598801266210458,0.4584936783554908]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>232</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>38</td>
                  
                  <td>17</td>
                  
                  <td>97</td>
                  
                  <td>43</td>
                  
                  <td>61</td>
                  
                  <td>1876</td>
                  
                  <td>107</td>
                  
                  <td>456945</td>
                  
                  <td>0.333</td>
                  
                  <td>0.667</td>
                  
                  <td>0.203</td>
                  
                  <td>0.212</td>
                  
                  <td>0.192</td>
                  
                  <td>0.272</td>
                  
                  <td>0.246</td>
                  
                  <td>0.229</td>
                  
                  <td>0.274</td>
                  
                  <td>0.228</td>
                  
                  <td>[0.0,0.0,0.333,0.667,0.203,0.212,0.192,0.272,0.246,0.229,0.274,0.228]</td>
                  
                  <td>[0.4062759484059352,-0.08324531030028531,-0.805903161333262]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>125</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>6</td>
                  
                  <td>1</td>
                  
                  <td>3</td>
                  
                  <td>3</td>
                  
                  <td>2</td>
                  
                  <td>62</td>
                  
                  <td>106</td>
                  
                  <td>15297</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.032</td>
                  
                  <td>0.013</td>
                  
                  <td>0.006</td>
                  
                  <td>0.019</td>
                  
                  <td>0.008</td>
                  
                  <td>0.007</td>
                  
                  <td>0.272</td>
                  
                  <td>0.007</td>
                  
                  <td>[1.0,0.0,0.0,0.0,0.032,0.013,0.006,0.019,0.008,0.007,0.272,0.007]</td>
                  
                  <td>[-0.08881119416490599,-0.9992685671470856,0.015478366508729292]</td>
                  
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
                  
                  <td>[0.9763470144395641,-0.08512521034479886,0.2779337409570378]</td>
                  
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
                  
                  <td>[0.9725252928055664,-0.0810439202393318,0.2943729965044828]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100022</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>3</td>
                  
                  <td>44</td>
                  
                  <td>18</td>
                  
                  <td>67</td>
                  
                  <td>20</td>
                  
                  <td>51</td>
                  
                  <td>1833</td>
                  
                  <td>66</td>
                  
                  <td>448366</td>
                  
                  <td>0.667</td>
                  
                  <td>1.0</td>
                  
                  <td>0.235</td>
                  
                  <td>0.225</td>
                  
                  <td>0.132</td>
                  
                  <td>0.127</td>
                  
                  <td>0.206</td>
                  
                  <td>0.224</td>
                  
                  <td>0.169</td>
                  
                  <td>0.224</td>
                  
                  <td>[0.0,1.0,0.667,1.0,0.235,0.225,0.132,0.127,0.206,0.224,0.169,0.224]</td>
                  
                  <td>[1.346932585630102,-0.15159147759788635,-0.5292432759118176]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>265</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>26</td>
                  
                  <td>2</td>
                  
                  <td>14</td>
                  
                  <td>12</td>
                  
                  <td>9</td>
                  
                  <td>389</td>
                  
                  <td>91</td>
                  
                  <td>102556</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.139</td>
                  
                  <td>0.025</td>
                  
                  <td>0.028</td>
                  
                  <td>0.076</td>
                  
                  <td>0.036</td>
                  
                  <td>0.047</td>
                  
                  <td>0.233</td>
                  
                  <td>0.051</td>
                  
                  <td>[0.0,0.0,0.0,0.0,0.139,0.025,0.028,0.076,0.036,0.047,0.233,0.051]</td>
                  
                  <td>[0.04223910912931981,-0.012914111821337802,-0.11465090457763114]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>33</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>9</td>
                  
                  <td>62</td>
                  
                  <td>32</td>
                  
                  <td>35</td>
                  
                  <td>1369</td>
                  
                  <td>162</td>
                  
                  <td>339373</td>
                  
                  <td>0.0</td>
                  
                  <td>0.333</td>
                  
                  <td>0.043</td>
                  
                  <td>0.113</td>
                  
                  <td>0.123</td>
                  
                  <td>0.203</td>
                  
                  <td>0.141</td>
                  
                  <td>0.167</td>
                  
                  <td>0.415</td>
                  
                  <td>0.169</td>
                  
                  <td>[1.0,1.0,0.0,0.333,0.043,0.113,0.123,0.203,0.141,0.167,0.415,0.169]</td>
                  
                  <td>[1.0082280690276773,-1.1071521756765845,0.09139604523728487]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100034</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>9</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>167</td>
                  
                  <td>48</td>
                  
                  <td>38964</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.005</td>
                  
                  <td>0.013</td>
                  
                  <td>0.018</td>
                  
                  <td>0.0</td>
                  
                  <td>0.02</td>
                  
                  <td>0.02</td>
                  
                  <td>0.123</td>
                  
                  <td>0.019</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.005,0.013,0.018,0.0,0.02,0.02,0.123,0.019]</td>
                  
                  <td>[0.8918080416404485,-0.06618259533852966,0.41676725023172057]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>95</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>21</td>
                  
                  <td>122</td>
                  
                  <td>38</td>
                  
                  <td>45</td>
                  
                  <td>2062</td>
                  
                  <td>59</td>
                  
                  <td>508151</td>
                  
                  <td>0.0</td>
                  
                  <td>0.0</td>
                  
                  <td>0.021</td>
                  
                  <td>0.263</td>
                  
                  <td>0.241</td>
                  
                  <td>0.241</td>
                  
                  <td>0.181</td>
                  
                  <td>0.252</td>
                  
                  <td>0.151</td>
                  
                  <td>0.254</td>
                  
                  <td>[0.0,1.0,0.0,0.0,0.021,0.263,0.241,0.241,0.181,0.252,0.151,0.254]</td>
                  
                  <td>[1.1098597232763217,-0.11286925805146947,0.023802952960169696]</td>
                  
                </tr>
                
                <tr>
                  
                  <td>18</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>30</td>
                  
                  <td>54</td>
                  
                  <td>246</td>
                  
                  <td>63</td>
                  
                  <td>112</td>
                  
                  <td>4640</td>
                  
                  <td>91</td>
                  
                  <td>1153807</td>
                  
                  <td>0.333</td>
                  
                  <td>0.333</td>
                  
                  <td>0.16</td>
                  
                  <td>0.675</td>
                  
                  <td>0.486</td>
                  
                  <td>0.399</td>
                  
                  <td>0.452</td>
                  
                  <td>0.567</td>
                  
                  <td>0.233</td>
                  
                  <td>0.576</td>
                  
                  <td>[1.0,1.0,0.333,0.333,0.16,0.675,0.486,0.399,0.452,0.567,0.233,0.576]</td>
                  
                  <td>[1.3936544424971644,-1.18443963622584,-0.6974450347200878]</td>
                  
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
                  
                  <td>[1.087966918654497,-1.1235960133832528,-0.16208492360100268]</td>
                  
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
      var tableWrapper = $('.df-table-wrapper-681284ee');
      var fixedHeader = $('.fixed-header', tableWrapper);
      var tableContainer = $('.df-table-container', tableWrapper);
      var table = $('.df-table', tableContainer);
      var rows = $('tbody > tr', table);
      var total = 100;
  
      fixedHeader
        .css('width', table.width())
        .find('.fixed-cell')
        .each(function(i, e) {
          $(this).css('width', $('.df-table-wrapper-681284ee th:nth-child(' + (i+1) + ')').css('width'));
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



![png](output_41_2.png)



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



![png](output_42_2.png)


## Heat map on feature and label correlation


```python
# print correlation between variables
corr = pd_features_df.drop("userId", axis=1).corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True));
```


![png](output_44_0.png)


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

    Explained variance by 3 principal components: 88.12%


## Save features as csv to avoid long loading times


```python
# save to csv 
features_df.toPandas().to_csv("features_pd_df.csv")
```

# Modeling

## Split in training, test, validation set


```python
train, test = features_df.randomSplit([0.8, 0.2], seed=42)

plt.hist(features_df.toPandas()['label'])
plt.show()
```


![png](output_54_0.png)


## Analyze label class imbalance


```python
# calculate balancing ratio for account for class imbalance
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
lr = LogisticRegression(featuresCol = 'pcaFeatures', labelCol = 'label', weightCol="classWeights")
```


```python
# fit training data to lr model and check performance before further refinement
lr_model = lr.fit(train)
training_summary = lr_model.summary
```


```python
# print precision and recall
pr = training_summary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()
```


![png](output_60_0.png)



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
simple_pred = lr_model.transform(test)
evaluate_prediction(simple_pred)
```

    F1 Score: 0.38, Recall: 0.62, Precision: 0.27





    (0.37681159420289856, 0.6190476190476191, 0.2708333333333333)




```python
# create evaluator
evaluator = BinaryClassificationEvaluator(metricName = 'areaUnderPR')

# tune model via CrossValidator and parameter Grid 
# build paramGrid
paramGrid = (ParamGridBuilder() \
    .addGrid(lr.maxIter, [1, 10, 100]) \
    .addGrid(lr.regParam,[0.0, 0.5, 2.0]) \
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
crossval_pred = crossval_model.transform(test)
```


```python
# get param maps of crossval_model
best_params = crossval_model.getEstimatorParamMaps()
```

## Model Evaluation

* use scikit learn metrics f1, precision, recall for model evaluation


```python
evaluate_prediction(crossval_pred)
```


```python
# evaluate results
#pred = crossval_pred
#pd_pred = pred.toPandas()

# show resulting format
#pd_pred.head(3)

# calculate score for f1, precision, recall
#f1 = f1_score(pd_pred.label, pd_pred.prediction)
#recall = recall_score(pd_pred.label, pd_pred.prediction)
#precision = precision_score(pd_pred.label, pd_pred.prediction)

#print("F1 Score: {:.2f}, Recall: {:.2f}, Precision: {:.2f}".format(f1, recall, precision))
```

    F1 Score: 0.00, Recall: 0.00, Precision: 0.00


    /opt/ibm/conda/miniconda3.6/lib/python3.6/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /opt/ibm/conda/miniconda3.6/lib/python3.6/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)


# Check Decision Tree as alternative estimator


```python
# Create a decision tree estimator object
dt = DecisionTreeClassifier(featuresCol = 'pcaFeatures', labelCol = 'label')
```


```python
# fit training data to lr model and check performance before further refinement
dt_model = dt.fit(train)
#dt_training_summary = dt_model.summary
```


```python
# print precision and recall
#dt_pr = dt_training_summary.pr.toPandas()
#plt.plot(dt_pr['recall'],dt_pr['precision'])
#plt.ylabel('Precision')
#plt.xlabel('Recall')
#plt.show()
```


```python
# transform testing data and check results
dt_simple_pred = dt_model.transform(test)
evaluate_prediction(dt_simple_pred)
```


```python
# tune dt model via CrossValidator and parameter Grid 
# build paramGrid
dt_paramGrid = (ParamGridBuilder() \
    .addGrid(dt.maxDepth, [2, 5, 8]) \
    .addGrid(dt.maxBins,[20, 50]) \
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
# predict on test data
dt_crossval_pred = dt_crossval_model.transform(test)

# get param maps of crossval_model
dt_best_params = dt_crossval_model.getEstimatorParamMaps()

#evaluate prediction results
evaluate_prediction(crossval_pred)
```
