
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
from pyspark.ml.classification import LogisticRegression

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
    <div id="chartFigure28f0c736" class="pd_save is-viewer-good" style="overflow-x:auto">
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
  
  
  <div class="df-table-wrapper df-table-wrapper-28f0c736 panel-group pd_save">
    <!-- dataframe schema -->
    
    <div class="panel panel-default">
      <div class="panel-heading">
        <h4 class="panel-title" style="margin: 0px;">
          <a data-toggle="collapse" href="#df-schema-28f0c736" data-parent="#df-table-wrapper-28f0c736">Schema</a>
        </h4>
      </div>
      <div id="df-schema-28f0c736" class="panel-collapse collapse">
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
            
              <div class="df-schema-field"><strong>registration: </strong> float64</div>
            
              <div class="df-schema-field"><strong>sessionId: </strong> int64</div>
            
              <div class="df-schema-field"><strong>song: </strong> object</div>
            
              <div class="df-schema-field"><strong>status: </strong> int64</div>
            
              <div class="df-schema-field"><strong>ts: </strong> int64</div>
            
              <div class="df-schema-field"><strong>userAgent: </strong> object</div>
            
              <div class="df-schema-field"><strong>userId: </strong> object</div>
            
          </div>
        </div>
      </div>
    </div>
    
    <!-- dataframe table -->
    <div class="panel panel-default">
      
      <div class="panel-heading">
        <h4 class="panel-title" style="margin: 0px;">
          <a data-toggle="collapse" href="#df-table-28f0c736" data-parent="#df-table-wrapper-28f0c736"> Table</a>
        </h4>
      </div>
      
      <div id="df-table-28f0c736" class="panel-collapse collapse in">
        <div class="panel-body">
          
          <input type="text" class="df-table-search form-control input-sm" placeholder="Search table">
          
          <div>
            
            <span class="df-table-search-count">Showing 100 of 543705 rows</span>
            
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
                  
                </tr>
              </thead>
              <tbody>
                
                <tr>
                  
                  <td>Bob Dylan</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>23</td>
                  
                  <td>Campos</td>
                  
                  <td>256.96608</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000.0</td>
                  
                  <td>245</td>
                  
                  <td>Simple Twist Of Fate</td>
                  
                  <td>200</td>
                  
                  <td>1538352318000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                </tr>
                
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
                  
                  <td>1538069638000.0</td>
                  
                  <td>97</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538352947000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>98</td>
                  
                </tr>
                
                <tr>
                  
                  <td>T.I.</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alexi</td>
                  
                  <td>F</td>
                  
                  <td>2</td>
                  
                  <td>Warren</td>
                  
                  <td>214.77832</td>
                  
                  <td>paid</td>
                  
                  <td>Spokane-Spokane Valley, WA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532482662000.0</td>
                  
                  <td>53</td>
                  
                  <td>Why You Wanna (Amended Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538354457000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:32.0) Gecko/20100101 Firefox/32.0</td>
                  
                  <td>54</td>
                  
                </tr>
                
                <tr>
                  
                  <td>A Perfect Circle</td>
                  
                  <td>Logged In</td>
                  
                  <td>Joseph</td>
                  
                  <td>M</td>
                  
                  <td>45</td>
                  
                  <td>Morales</td>
                  
                  <td>336.48281</td>
                  
                  <td>free</td>
                  
                  <td>Corpus Christi, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532063507000.0</td>
                  
                  <td>292</td>
                  
                  <td>Counting Bodies Like Sheep To The Rhythm Of The War Drums (Explicit Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538355807000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>293</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Toy-Box</td>
                  
                  <td>Logged In</td>
                  
                  <td>Maverick</td>
                  
                  <td>M</td>
                  
                  <td>211</td>
                  
                  <td>Santiago</td>
                  
                  <td>217.02485</td>
                  
                  <td>paid</td>
                  
                  <td>Orlando-Kissimmee-Sanford, FL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535953455000.0</td>
                  
                  <td>178</td>
                  
                  <td>Earth_ Wind_ Water &amp; Fire</td>
                  
                  <td>200</td>
                  
                  <td>1538357005000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>179</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Brad Paisley With Andy Griffith</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>9</td>
                  
                  <td>Hogan</td>
                  
                  <td>302.70649</td>
                  
                  <td>free</td>
                  
                  <td>Denver-Aurora-Lakewood, CO</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535066380000.0</td>
                  
                  <td>100</td>
                  
                  <td>Waitin' On A Woman</td>
                  
                  <td>200</td>
                  
                  <td>1538360408000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.2; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>101</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Anthony</td>
                  
                  <td>M</td>
                  
                  <td>27</td>
                  
                  <td>Reed</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Miami-Fort Lauderdale-West Palm Beach, FL</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1534823030000.0</td>
                  
                  <td>511</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538364430000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>166</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Erin Bode</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ashlynn</td>
                  
                  <td>F</td>
                  
                  <td>33</td>
                  
                  <td>Williams</td>
                  
                  <td>252.05506</td>
                  
                  <td>free</td>
                  
                  <td>Tallahassee, FL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537365219000.0</td>
                  
                  <td>427</td>
                  
                  <td>Here_ There And Everywhere</td>
                  
                  <td>200</td>
                  
                  <td>1538365423000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>74</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Soundgarden</td>
                  
                  <td>Logged In</td>
                  
                  <td>Maverick</td>
                  
                  <td>M</td>
                  
                  <td>249</td>
                  
                  <td>Santiago</td>
                  
                  <td>290.11546</td>
                  
                  <td>paid</td>
                  
                  <td>Orlando-Kissimmee-Sanford, FL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535953455000.0</td>
                  
                  <td>178</td>
                  
                  <td>Burden In My Hand</td>
                  
                  <td>200</td>
                  
                  <td>1538365457000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>179</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Five Iron Frenzy</td>
                  
                  <td>Logged In</td>
                  
                  <td>Carter</td>
                  
                  <td>M</td>
                  
                  <td>31</td>
                  
                  <td>Cook</td>
                  
                  <td>236.09424</td>
                  
                  <td>paid</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1522793334000.0</td>
                  
                  <td>287</td>
                  
                  <td>Canada</td>
                  
                  <td>200</td>
                  
                  <td>1538366146000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>288</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Amos Lee</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>27</td>
                  
                  <td>Roberts</td>
                  
                  <td>169.79546</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537634865000.0</td>
                  
                  <td>27</td>
                  
                  <td>Better Days</td>
                  
                  <td>200</td>
                  
                  <td>1538367062000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Caleb</td>
                  
                  <td>M</td>
                  
                  <td>28</td>
                  
                  <td>Lane</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>Add Friend</td>
                  
                  <td>1536756625000.0</td>
                  
                  <td>281</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538369176000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>282</td>
                  
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
                  
                  <td>1535953455000.0</td>
                  
                  <td>178</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538369208000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>179</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Destiny's Child</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>38</td>
                  
                  <td>Roberts</td>
                  
                  <td>271.33342</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537634865000.0</td>
                  
                  <td>27</td>
                  
                  <td>Say My Name</td>
                  
                  <td>200</td>
                  
                  <td>1538369299000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Shinedown</td>
                  
                  <td>Logged In</td>
                  
                  <td>Adriel</td>
                  
                  <td>M</td>
                  
                  <td>2</td>
                  
                  <td>Mendoza</td>
                  
                  <td>233.97832</td>
                  
                  <td>paid</td>
                  
                  <td>Kansas City, MO-KS</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535623466000.0</td>
                  
                  <td>476</td>
                  
                  <td>Sound Of Madness (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538369966000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"</td>
                  
                  <td>18</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Warren Zevon</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>119</td>
                  
                  <td>Campos</td>
                  
                  <td>201.63873</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000.0</td>
                  
                  <td>245</td>
                  
                  <td>Mr. Bad Example</td>
                  
                  <td>200</td>
                  
                  <td>1538370448000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Jack Johnson</td>
                  
                  <td>Logged In</td>
                  
                  <td>Aurora</td>
                  
                  <td>F</td>
                  
                  <td>45</td>
                  
                  <td>Humphrey</td>
                  
                  <td>137.50812</td>
                  
                  <td>paid</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536795126000.0</td>
                  
                  <td>418</td>
                  
                  <td>We're Going To Be Friends</td>
                  
                  <td>200</td>
                  
                  <td>1538373179000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>127</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Trews</td>
                  
                  <td>Logged In</td>
                  
                  <td>Carter</td>
                  
                  <td>M</td>
                  
                  <td>67</td>
                  
                  <td>Cook</td>
                  
                  <td>200.80281</td>
                  
                  <td>paid</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1522793334000.0</td>
                  
                  <td>287</td>
                  
                  <td>I Can't Stop Laughing</td>
                  
                  <td>200</td>
                  
                  <td>1538373436000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>288</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Train</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lucas</td>
                  
                  <td>M</td>
                  
                  <td>90</td>
                  
                  <td>Decker</td>
                  
                  <td>205.45261</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534945722000.0</td>
                  
                  <td>222</td>
                  
                  <td>Marry Me</td>
                  
                  <td>200</td>
                  
                  <td>1538373696000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>223</td>
                  
                </tr>
                
                <tr>
                  
                  <td>J-Kwon</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jhaden</td>
                  
                  <td>M</td>
                  
                  <td>61</td>
                  
                  <td>Cooper</td>
                  
                  <td>243.01669</td>
                  
                  <td>free</td>
                  
                  <td>Knoxville, TN</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536859413000.0</td>
                  
                  <td>249</td>
                  
                  <td>Tipsy</td>
                  
                  <td>200</td>
                  
                  <td>1538374479000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>250</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Erasure</td>
                  
                  <td>Logged In</td>
                  
                  <td>Faigy</td>
                  
                  <td>F</td>
                  
                  <td>6</td>
                  
                  <td>Howe</td>
                  
                  <td>236.45995</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538211832000.0</td>
                  
                  <td>492</td>
                  
                  <td>Love To Hate You</td>
                  
                  <td>200</td>
                  
                  <td>1538374528000</td>
                  
                  <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>95</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Faigy</td>
                  
                  <td>F</td>
                  
                  <td>7</td>
                  
                  <td>Howe</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1538211832000.0</td>
                  
                  <td>492</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538374529000</td>
                  
                  <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>95</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Justin Timberlake</td>
                  
                  <td>Logged In</td>
                  
                  <td>Grant</td>
                  
                  <td>M</td>
                  
                  <td>22</td>
                  
                  <td>Flores</td>
                  
                  <td>277.9424</td>
                  
                  <td>free</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538120859000.0</td>
                  
                  <td>141</td>
                  
                  <td>LoveStoned/I Think She Knows</td>
                  
                  <td>200</td>
                  
                  <td>1538375188000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>142</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Daft Punk</td>
                  
                  <td>Logged In</td>
                  
                  <td>Aurora</td>
                  
                  <td>F</td>
                  
                  <td>63</td>
                  
                  <td>Humphrey</td>
                  
                  <td>418.55955</td>
                  
                  <td>paid</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536795126000.0</td>
                  
                  <td>418</td>
                  
                  <td>Face To Face (Demon Remix)</td>
                  
                  <td>200</td>
                  
                  <td>1538375189000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>127</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Mariah Carey Featuring Busta Rhymes_ Fabulous And  DJ Clue</td>
                  
                  <td>Logged In</td>
                  
                  <td>Carter</td>
                  
                  <td>M</td>
                  
                  <td>78</td>
                  
                  <td>Cook</td>
                  
                  <td>403.35628</td>
                  
                  <td>paid</td>
                  
                  <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1522793334000.0</td>
                  
                  <td>287</td>
                  
                  <td>Last Night A DJ Saved My Life</td>
                  
                  <td>200</td>
                  
                  <td>1538375811000</td>
                  
                  <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>288</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Rosana</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jessiah</td>
                  
                  <td>M</td>
                  
                  <td>17</td>
                  
                  <td>Rose</td>
                  
                  <td>282.43546</td>
                  
                  <td>free</td>
                  
                  <td>Richmond, VA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532413080000.0</td>
                  
                  <td>529</td>
                  
                  <td>SoÃÂ±are</td>
                  
                  <td>200</td>
                  
                  <td>1538375844000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>207</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Opio</td>
                  
                  <td>Logged In</td>
                  
                  <td>Adriel</td>
                  
                  <td>M</td>
                  
                  <td>43</td>
                  
                  <td>Mendoza</td>
                  
                  <td>168.56771</td>
                  
                  <td>paid</td>
                  
                  <td>Kansas City, MO-KS</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535623466000.0</td>
                  
                  <td>476</td>
                  
                  <td>The Grassy Knoll</td>
                  
                  <td>200</td>
                  
                  <td>1538377774000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"</td>
                  
                  <td>18</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jhaden</td>
                  
                  <td>M</td>
                  
                  <td>83</td>
                  
                  <td>Cooper</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Knoxville, TN</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1536859413000.0</td>
                  
                  <td>249</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538378466000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>250</td>
                  
                </tr>
                
                <tr>
                  
                  <td>SUPREME BEINGS OF LEISURE</td>
                  
                  <td>Logged In</td>
                  
                  <td>Adriel</td>
                  
                  <td>M</td>
                  
                  <td>53</td>
                  
                  <td>Mendoza</td>
                  
                  <td>206.34077</td>
                  
                  <td>paid</td>
                  
                  <td>Kansas City, MO-KS</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535623466000.0</td>
                  
                  <td>476</td>
                  
                  <td>This World (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538379721000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"</td>
                  
                  <td>18</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Crosby_ Stills_ Nash and Young</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>11</td>
                  
                  <td>Taylor</td>
                  
                  <td>39.60118</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533764798000.0</td>
                  
                  <td>522</td>
                  
                  <td>Woodstock</td>
                  
                  <td>200</td>
                  
                  <td>1538381885000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>195</td>
                  
                  <td>Campos</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Down</td>
                  
                  <td>1535931018000.0</td>
                  
                  <td>245</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538382975000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Langtry and the Pocket-Sized Planets</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>26</td>
                  
                  <td>Taylor</td>
                  
                  <td>272.79628</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533764798000.0</td>
                  
                  <td>522</td>
                  
                  <td>Sampler</td>
                  
                  <td>200</td>
                  
                  <td>1538385033000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Knife</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>29</td>
                  
                  <td>Taylor</td>
                  
                  <td>292.54485</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533764798000.0</td>
                  
                  <td>522</td>
                  
                  <td>Silent Shout</td>
                  
                  <td>200</td>
                  
                  <td>1538385441000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
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
                  
                  <td>1536859413000.0</td>
                  
                  <td>249</td>
                  
                  <td>Escape To Paradise</td>
                  
                  <td>200</td>
                  
                  <td>1538385589000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>250</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Evan</td>
                  
                  <td>M</td>
                  
                  <td>2</td>
                  
                  <td>Shelton</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Hagerstown-Martinsburg, MD-WV</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1534894284000.0</td>
                  
                  <td>479</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538386563000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>251</td>
                  
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
                  
                  <td>1537634865000.0</td>
                  
                  <td>27</td>
                  
                  <td>Sehr kosmisch</td>
                  
                  <td>200</td>
                  
                  <td>1538389869000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jackson</td>
                  
                  <td>M</td>
                  
                  <td>5</td>
                  
                  <td>Hoffman</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1537054964000.0</td>
                  
                  <td>184</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538389990000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>185</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Justin Bieber</td>
                  
                  <td>Logged In</td>
                  
                  <td>Teagan</td>
                  
                  <td>F</td>
                  
                  <td>147</td>
                  
                  <td>Roberts</td>
                  
                  <td>220.89098</td>
                  
                  <td>free</td>
                  
                  <td>New Philadelphia-Dover, OH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537634865000.0</td>
                  
                  <td>27</td>
                  
                  <td>Somebody To Love</td>
                  
                  <td>200</td>
                  
                  <td>1538390760000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>28</td>
                  
                </tr>
                
                <tr>
                  
                  <td>La Shica</td>
                  
                  <td>Logged In</td>
                  
                  <td>Emily</td>
                  
                  <td>F</td>
                  
                  <td>13</td>
                  
                  <td>Williams</td>
                  
                  <td>245.68118</td>
                  
                  <td>free</td>
                  
                  <td>Austin-Round Rock, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536852701000.0</td>
                  
                  <td>172</td>
                  
                  <td>Madre</td>
                  
                  <td>200</td>
                  
                  <td>1538391186000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>173</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Evan</td>
                  
                  <td>M</td>
                  
                  <td>35</td>
                  
                  <td>Shelton</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Hagerstown-Martinsburg, MD-WV</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1534894284000.0</td>
                  
                  <td>479</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538392399000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>251</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Dam Funk</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jackson</td>
                  
                  <td>M</td>
                  
                  <td>19</td>
                  
                  <td>Hoffman</td>
                  
                  <td>199.05261</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537054964000.0</td>
                  
                  <td>184</td>
                  
                  <td>Mirrors</td>
                  
                  <td>200</td>
                  
                  <td>1538392832000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>185</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Brayden</td>
                  
                  <td>M</td>
                  
                  <td>30</td>
                  
                  <td>Thomas</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1534133898000.0</td>
                  
                  <td>498</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538393595000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>85</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged Out</td>
                  
                  <td>None</td>
                  
                  <td>None</td>
                  
                  <td>34</td>
                  
                  <td>None</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>None</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>nan</td>
                  
                  <td>498</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538394116000</td>
                  
                  <td>None</td>
                  
                  <td></td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Payton</td>
                  
                  <td>F</td>
                  
                  <td>12</td>
                  
                  <td>Campbell</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1529027541000.0</td>
                  
                  <td>518</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538394715000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"</td>
                  
                  <td>39</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Daelin</td>
                  
                  <td>M</td>
                  
                  <td>46</td>
                  
                  <td>Turner</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1538227408000.0</td>
                  
                  <td>125</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538395085000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>126</td>
                  
                </tr>
                
                <tr>
                  
                  <td>RIP SLYME</td>
                  
                  <td>Logged In</td>
                  
                  <td>Faigy</td>
                  
                  <td>F</td>
                  
                  <td>120</td>
                  
                  <td>Howe</td>
                  
                  <td>283.21914</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538211832000.0</td>
                  
                  <td>492</td>
                  
                  <td>FUNKASTIC</td>
                  
                  <td>200</td>
                  
                  <td>1538398172000</td>
                  
                  <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>95</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Belle &amp; Sebastian</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>9</td>
                  
                  <td>Taylor</td>
                  
                  <td>157.70077</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533764798000.0</td>
                  
                  <td>547</td>
                  
                  <td>(I Believe In) Travellin' Light</td>
                  
                  <td>200</td>
                  
                  <td>1538399579000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>11</td>
                  
                  <td>Taylor</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1533764798000.0</td>
                  
                  <td>547</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538400080000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Faigy</td>
                  
                  <td>F</td>
                  
                  <td>133</td>
                  
                  <td>Howe</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>Logout</td>
                  
                  <td>1538211832000.0</td>
                  
                  <td>492</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538400628000</td>
                  
                  <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>95</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>7</td>
                  
                  <td>Raymond</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Hartford-West Hartford-East Hartford, CT</td>
                  
                  <td>GET</td>
                  
                  <td>Upgrade</td>
                  
                  <td>1534245996000.0</td>
                  
                  <td>26</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538401345000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Ordinary Boys</td>
                  
                  <td>Logged In</td>
                  
                  <td>Payton</td>
                  
                  <td>F</td>
                  
                  <td>47</td>
                  
                  <td>Campbell</td>
                  
                  <td>162.87302</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529027541000.0</td>
                  
                  <td>518</td>
                  
                  <td>Boys Will Be Boys</td>
                  
                  <td>200</td>
                  
                  <td>1538401359000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"</td>
                  
                  <td>39</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Beanfield feat. Bajka</td>
                  
                  <td>Logged In</td>
                  
                  <td>Faigy</td>
                  
                  <td>F</td>
                  
                  <td>152</td>
                  
                  <td>Howe</td>
                  
                  <td>594.20689</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538211832000.0</td>
                  
                  <td>492</td>
                  
                  <td>Tides (C's Movement #1)</td>
                  
                  <td>200</td>
                  
                  <td>1538403625000</td>
                  
                  <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>95</td>
                  
                </tr>
                
                <tr>
                  
                  <td>ARRESTED DEVELOPMENT</td>
                  
                  <td>Logged In</td>
                  
                  <td>Molly</td>
                  
                  <td>F</td>
                  
                  <td>11</td>
                  
                  <td>Harrison</td>
                  
                  <td>200.98567</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534255113000.0</td>
                  
                  <td>142</td>
                  
                  <td>Fountain Of Youth</td>
                  
                  <td>200</td>
                  
                  <td>1538404041000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>143</td>
                  
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
                  
                  <td>1534245996000.0</td>
                  
                  <td>26</td>
                  
                  <td>The New Black</td>
                  
                  <td>200</td>
                  
                  <td>1538406621000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Gerbils</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>35</td>
                  
                  <td>Raymond</td>
                  
                  <td>27.01016</td>
                  
                  <td>free</td>
                  
                  <td>Hartford-West Hartford-East Hartford, CT</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534245996000.0</td>
                  
                  <td>26</td>
                  
                  <td>(iii)</td>
                  
                  <td>200</td>
                  
                  <td>1538407271000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Avenged Sevenfold</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lucero</td>
                  
                  <td>F</td>
                  
                  <td>10</td>
                  
                  <td>Reed</td>
                  
                  <td>333.13914</td>
                  
                  <td>free</td>
                  
                  <td>Louisville/Jefferson County, KY-IN</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536642109000.0</td>
                  
                  <td>139</td>
                  
                  <td>Seize The Day (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538407295000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>140</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Eva Cassidy</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>39</td>
                  
                  <td>Raymond</td>
                  
                  <td>146.28526</td>
                  
                  <td>free</td>
                  
                  <td>Hartford-West Hartford-East Hartford, CT</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534245996000.0</td>
                  
                  <td>26</td>
                  
                  <td>Way Beyond The Blue (Album Version)</td>
                  
                  <td>200</td>
                  
                  <td>1538407948000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Kent</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>41</td>
                  
                  <td>Raymond</td>
                  
                  <td>407.71873</td>
                  
                  <td>free</td>
                  
                  <td>Hartford-West Hartford-East Hartford, CT</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534245996000.0</td>
                  
                  <td>26</td>
                  
                  <td>Vals fÃÂ¶r satan (din vÃÂ¤n pessimisten)</td>
                  
                  <td>200</td>
                  
                  <td>1538408373000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Bryson</td>
                  
                  <td>M</td>
                  
                  <td>17</td>
                  
                  <td>Roberson</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Houston-The Woodlands-Sugar Land, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>Add to Playlist</td>
                  
                  <td>1521380675000.0</td>
                  
                  <td>5</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538409683000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>6</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Evanescence</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>46</td>
                  
                  <td>Raymond</td>
                  
                  <td>236.12036</td>
                  
                  <td>free</td>
                  
                  <td>Hartford-West Hartford-East Hartford, CT</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534245996000.0</td>
                  
                  <td>26</td>
                  
                  <td>Bring Me To Life</td>
                  
                  <td>200</td>
                  
                  <td>1538409955000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                </tr>
                
                <tr>
                  
                  <td>La Rue Ketanou</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>68</td>
                  
                  <td>Campos</td>
                  
                  <td>205.08689</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000.0</td>
                  
                  <td>548</td>
                  
                  <td>Les derniers aventuriers</td>
                  
                  <td>200</td>
                  
                  <td>1538410390000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Morgan</td>
                  
                  <td>F</td>
                  
                  <td>2</td>
                  
                  <td>Reid</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Boston-Cambridge-Newton, MA-NH</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1537263152000.0</td>
                  
                  <td>441</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538410573000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>274</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Biosphere</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>69</td>
                  
                  <td>Campos</td>
                  
                  <td>189.85751</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000.0</td>
                  
                  <td>548</td>
                  
                  <td>Two Ocean Plateau</td>
                  
                  <td>200</td>
                  
                  <td>1538410595000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Miles Davis</td>
                  
                  <td>Logged In</td>
                  
                  <td>Bryson</td>
                  
                  <td>M</td>
                  
                  <td>22</td>
                  
                  <td>Roberson</td>
                  
                  <td>627.9571</td>
                  
                  <td>free</td>
                  
                  <td>Houston-The Woodlands-Sugar Land, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1521380675000.0</td>
                  
                  <td>5</td>
                  
                  <td>Basin Street Blues</td>
                  
                  <td>200</td>
                  
                  <td>1538410778000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>6</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Bryson</td>
                  
                  <td>M</td>
                  
                  <td>23</td>
                  
                  <td>Roberson</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Houston-The Woodlands-Sugar Land, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1521380675000.0</td>
                  
                  <td>5</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538410779000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>6</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Heartsrevolution</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>80</td>
                  
                  <td>Campos</td>
                  
                  <td>147.59138</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000.0</td>
                  
                  <td>548</td>
                  
                  <td>????????</td>
                  
                  <td>200</td>
                  
                  <td>1538411733000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Spor</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nicole</td>
                  
                  <td>F</td>
                  
                  <td>24</td>
                  
                  <td>Beck</td>
                  
                  <td>281.10322</td>
                  
                  <td>paid</td>
                  
                  <td>Vineland-Bridgeton, NJ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532224335000.0</td>
                  
                  <td>123</td>
                  
                  <td>Knock You Down</td>
                  
                  <td>200</td>
                  
                  <td>1538412528000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Larry Norman</td>
                  
                  <td>Logged In</td>
                  
                  <td>Bryson</td>
                  
                  <td>M</td>
                  
                  <td>32</td>
                  
                  <td>Roberson</td>
                  
                  <td>364.45995</td>
                  
                  <td>free</td>
                  
                  <td>Houston-The Woodlands-Sugar Land, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1521380675000.0</td>
                  
                  <td>5</td>
                  
                  <td>I Am the Six O'Clock News</td>
                  
                  <td>200</td>
                  
                  <td>1538412857000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>6</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Scooter</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>60</td>
                  
                  <td>Raymond</td>
                  
                  <td>235.4673</td>
                  
                  <td>free</td>
                  
                  <td>Hartford-West Hartford-East Hartford, CT</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534245996000.0</td>
                  
                  <td>26</td>
                  
                  <td>The Logical Song</td>
                  
                  <td>200</td>
                  
                  <td>1538413008000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                </tr>
                
                <tr>
                  
                  <td>BjÃÂ¶rk</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nicole</td>
                  
                  <td>F</td>
                  
                  <td>30</td>
                  
                  <td>Beck</td>
                  
                  <td>348.57751</td>
                  
                  <td>paid</td>
                  
                  <td>Vineland-Bridgeton, NJ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532224335000.0</td>
                  
                  <td>123</td>
                  
                  <td>Undo</td>
                  
                  <td>200</td>
                  
                  <td>1538413887000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Madelyn</td>
                  
                  <td>F</td>
                  
                  <td>7</td>
                  
                  <td>Henson</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Charlotte-Concord-Gastonia, NC-SC</td>
                  
                  <td>PUT</td>
                  
                  <td>Save Settings</td>
                  
                  <td>1532920994000.0</td>
                  
                  <td>112</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538414283000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>113</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Simon Harris</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lakyla</td>
                  
                  <td>F</td>
                  
                  <td>51</td>
                  
                  <td>Porter</td>
                  
                  <td>195.83955</td>
                  
                  <td>paid</td>
                  
                  <td>Modesto, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535849930000.0</td>
                  
                  <td>507</td>
                  
                  <td>Sample Track 2</td>
                  
                  <td>200</td>
                  
                  <td>1538414493000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>162</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Lily Allen</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lucero</td>
                  
                  <td>F</td>
                  
                  <td>23</td>
                  
                  <td>Reed</td>
                  
                  <td>185.25995</td>
                  
                  <td>free</td>
                  
                  <td>Louisville/Jefferson County, KY-IN</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536642109000.0</td>
                  
                  <td>570</td>
                  
                  <td>22</td>
                  
                  <td>200</td>
                  
                  <td>1538416865000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>140</td>
                  
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
                  
                  <td>1532224335000.0</td>
                  
                  <td>123</td>
                  
                  <td>You'd Be So Nice To Come Home To</td>
                  
                  <td>200</td>
                  
                  <td>1538417366000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Jason Aldean</td>
                  
                  <td>Logged In</td>
                  
                  <td>Kee</td>
                  
                  <td>M</td>
                  
                  <td>30</td>
                  
                  <td>Taylor</td>
                  
                  <td>203.75465</td>
                  
                  <td>free</td>
                  
                  <td>Virginia Beach-Norfolk-Newport News, VA-NC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1533764798000.0</td>
                  
                  <td>562</td>
                  
                  <td>Big Green Tractor</td>
                  
                  <td>200</td>
                  
                  <td>1538417377000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>196</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Erin McKeown</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jasmine</td>
                  
                  <td>F</td>
                  
                  <td>27</td>
                  
                  <td>Richardson</td>
                  
                  <td>338.46812</td>
                  
                  <td>free</td>
                  
                  <td>Philadelphia-Camden-Wilmington, PA-NJ-DE-MD</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1531477817000.0</td>
                  
                  <td>166</td>
                  
                  <td>Fast As I Can</td>
                  
                  <td>200</td>
                  
                  <td>1538417600000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>167</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Coldplay</td>
                  
                  <td>Logged In</td>
                  
                  <td>Emily</td>
                  
                  <td>F</td>
                  
                  <td>92</td>
                  
                  <td>Morrison</td>
                  
                  <td>307.51302</td>
                  
                  <td>free</td>
                  
                  <td>San Francisco-Oakland-Hayward, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534367797000.0</td>
                  
                  <td>477</td>
                  
                  <td>Clocks</td>
                  
                  <td>200</td>
                  
                  <td>1538419482000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>232</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged Out</td>
                  
                  <td>None</td>
                  
                  <td>None</td>
                  
                  <td>9</td>
                  
                  <td>None</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>None</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>nan</td>
                  
                  <td>171</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538420260000</td>
                  
                  <td>None</td>
                  
                  <td></td>
                  
                </tr>
                
                <tr>
                  
                  <td>Aerosmith</td>
                  
                  <td>Logged In</td>
                  
                  <td>Gianna</td>
                  
                  <td>F</td>
                  
                  <td>122</td>
                  
                  <td>Campos</td>
                  
                  <td>294.922</td>
                  
                  <td>paid</td>
                  
                  <td>Mobile, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535931018000.0</td>
                  
                  <td>548</td>
                  
                  <td>Dream On</td>
                  
                  <td>200</td>
                  
                  <td>1538421131000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>246</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Thomas Newman</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nicole</td>
                  
                  <td>F</td>
                  
                  <td>61</td>
                  
                  <td>Beck</td>
                  
                  <td>104.07138</td>
                  
                  <td>paid</td>
                  
                  <td>Vineland-Bridgeton, NJ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532224335000.0</td>
                  
                  <td>123</td>
                  
                  <td>Cold Lamb Sandwich</td>
                  
                  <td>200</td>
                  
                  <td>1538421368000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Adal</td>
                  
                  <td>M</td>
                  
                  <td>15</td>
                  
                  <td>Murphy</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1536977188000.0</td>
                  
                  <td>275</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538422409000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/6.1.5 Safari/537.77.4"</td>
                  
                  <td>276</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Old Crow Medicine Show</td>
                  
                  <td>Logged In</td>
                  
                  <td>Spencer</td>
                  
                  <td>M</td>
                  
                  <td>31</td>
                  
                  <td>Gonzalez</td>
                  
                  <td>183.43138</td>
                  
                  <td>free</td>
                  
                  <td>Concord, NH</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537347211000.0</td>
                  
                  <td>64</td>
                  
                  <td>Bobcat Tracks</td>
                  
                  <td>200</td>
                  
                  <td>1538422502000</td>
                  
                  <td>Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>65</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Violent Femmes</td>
                  
                  <td>Logged In</td>
                  
                  <td>Madelyn</td>
                  
                  <td>F</td>
                  
                  <td>50</td>
                  
                  <td>Henson</td>
                  
                  <td>176.74404</td>
                  
                  <td>free</td>
                  
                  <td>Charlotte-Concord-Gastonia, NC-SC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532920994000.0</td>
                  
                  <td>112</td>
                  
                  <td>Kiss Off</td>
                  
                  <td>200</td>
                  
                  <td>1538423313000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>113</td>
                  
                </tr>
                
                <tr>
                  
                  <td>DAVE MATTHEWS BAND</td>
                  
                  <td>Logged In</td>
                  
                  <td>Jaleel</td>
                  
                  <td>M</td>
                  
                  <td>7</td>
                  
                  <td>Maldonado</td>
                  
                  <td>309.49832</td>
                  
                  <td>free</td>
                  
                  <td>Boulder, CO</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537890437000.0</td>
                  
                  <td>407</td>
                  
                  <td>One Sweet World</td>
                  
                  <td>200</td>
                  
                  <td>1538424794000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>59</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Kings Of Leon</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lakyla</td>
                  
                  <td>F</td>
                  
                  <td>96</td>
                  
                  <td>Porter</td>
                  
                  <td>181.13261</td>
                  
                  <td>paid</td>
                  
                  <td>Modesto, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535849930000.0</td>
                  
                  <td>507</td>
                  
                  <td>Ragoo</td>
                  
                  <td>200</td>
                  
                  <td>1538425432000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>162</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Shawn Colvin</td>
                  
                  <td>Logged In</td>
                  
                  <td>Madelyn</td>
                  
                  <td>F</td>
                  
                  <td>63</td>
                  
                  <td>Henson</td>
                  
                  <td>149.9424</td>
                  
                  <td>free</td>
                  
                  <td>Charlotte-Concord-Gastonia, NC-SC</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532920994000.0</td>
                  
                  <td>112</td>
                  
                  <td>Words</td>
                  
                  <td>200</td>
                  
                  <td>1538426060000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>113</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Alliance Ethnik</td>
                  
                  <td>Logged In</td>
                  
                  <td>Chase</td>
                  
                  <td>M</td>
                  
                  <td>45</td>
                  
                  <td>Ross</td>
                  
                  <td>195.94404</td>
                  
                  <td>free</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532450666000.0</td>
                  
                  <td>136</td>
                  
                  <td>SinceritÃÂ© Et Jalousie</td>
                  
                  <td>200</td>
                  
                  <td>1538426073000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.78.2 (KHTML, like Gecko) Version/7.0.6 Safari/537.78.2"</td>
                  
                  <td>137</td>
                  
                </tr>
                
                <tr>
                  
                  <td>ATB</td>
                  
                  <td>Logged In</td>
                  
                  <td>Erick</td>
                  
                  <td>M</td>
                  
                  <td>4</td>
                  
                  <td>Brooks</td>
                  
                  <td>191.7122</td>
                  
                  <td>free</td>
                  
                  <td>Selma, AL</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1537956751000.0</td>
                  
                  <td>57</td>
                  
                  <td>Rising Moon</td>
                  
                  <td>200</td>
                  
                  <td>1538426807000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>58</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Talk Talk</td>
                  
                  <td>Logged In</td>
                  
                  <td>Lakyla</td>
                  
                  <td>F</td>
                  
                  <td>107</td>
                  
                  <td>Porter</td>
                  
                  <td>317.75302</td>
                  
                  <td>paid</td>
                  
                  <td>Modesto, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535849930000.0</td>
                  
                  <td>507</td>
                  
                  <td>Give It Up</td>
                  
                  <td>200</td>
                  
                  <td>1538427045000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>162</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Keith Jarrett_ Gary Peacock_ Jack DeJohnette</td>
                  
                  <td>Logged In</td>
                  
                  <td>Brayden</td>
                  
                  <td>M</td>
                  
                  <td>23</td>
                  
                  <td>Thomas</td>
                  
                  <td>422.97424</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1534133898000.0</td>
                  
                  <td>556</td>
                  
                  <td>La Valse Bleue</td>
                  
                  <td>200</td>
                  
                  <td>1538428831000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>85</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Black Keys</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>200</td>
                  
                  <td>Myers</td>
                  
                  <td>145.65832</td>
                  
                  <td>paid</td>
                  
                  <td>Grand Rapids-Wyoming, MI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529995579000.0</td>
                  
                  <td>235</td>
                  
                  <td>Run Me Down</td>
                  
                  <td>200</td>
                  
                  <td>1538428865000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>236</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Gaz Nevada</td>
                  
                  <td>Logged In</td>
                  
                  <td>Arianna</td>
                  
                  <td>F</td>
                  
                  <td>10</td>
                  
                  <td>Bullock</td>
                  
                  <td>390.21669</td>
                  
                  <td>free</td>
                  
                  <td>Topeka, KS</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1538314334000.0</td>
                  
                  <td>282</td>
                  
                  <td>I C Love Affair</td>
                  
                  <td>200</td>
                  
                  <td>1538429786000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko</td>
                  
                  <td>283</td>
                  
                </tr>
                
                <tr>
                  
                  <td>John Mellencamp</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nicole</td>
                  
                  <td>F</td>
                  
                  <td>105</td>
                  
                  <td>Beck</td>
                  
                  <td>255.65995</td>
                  
                  <td>paid</td>
                  
                  <td>Vineland-Bridgeton, NJ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532224335000.0</td>
                  
                  <td>123</td>
                  
                  <td>Jack &amp; Diane</td>
                  
                  <td>200</td>
                  
                  <td>1538430384000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                </tr>
                
                <tr>
                  
                  <td>The Crystals</td>
                  
                  <td>Logged In</td>
                  
                  <td>Aurora</td>
                  
                  <td>F</td>
                  
                  <td>29</td>
                  
                  <td>Humphrey</td>
                  
                  <td>151.30077</td>
                  
                  <td>paid</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536795126000.0</td>
                  
                  <td>537</td>
                  
                  <td>Then He Kissed Me</td>
                  
                  <td>200</td>
                  
                  <td>1538432544000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>127</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Diam's</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>34</td>
                  
                  <td>Hogan</td>
                  
                  <td>250.72281</td>
                  
                  <td>free</td>
                  
                  <td>Denver-Aurora-Lakewood, CO</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535066380000.0</td>
                  
                  <td>523</td>
                  
                  <td>Dans Ma Bulle (Edit Radio - Live 2006)</td>
                  
                  <td>200</td>
                  
                  <td>1538433587000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.2; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>101</td>
                  
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
                  
                  <td>1534133898000.0</td>
                  
                  <td>556</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538434037000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>85</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Kermit Ruffins</td>
                  
                  <td>Logged In</td>
                  
                  <td>Nicole</td>
                  
                  <td>F</td>
                  
                  <td>125</td>
                  
                  <td>Beck</td>
                  
                  <td>349.83138</td>
                  
                  <td>paid</td>
                  
                  <td>Vineland-Bridgeton, NJ</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1532224335000.0</td>
                  
                  <td>123</td>
                  
                  <td>Skokiaan</td>
                  
                  <td>200</td>
                  
                  <td>1538434095000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>124</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Trivium</td>
                  
                  <td>Logged In</td>
                  
                  <td>Everett</td>
                  
                  <td>M</td>
                  
                  <td>90</td>
                  
                  <td>Quinn</td>
                  
                  <td>293.66812</td>
                  
                  <td>free</td>
                  
                  <td>Appleton, WI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1536082261000.0</td>
                  
                  <td>553</td>
                  
                  <td>Requiem</td>
                  
                  <td>200</td>
                  
                  <td>1538435984000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.94 Safari/537.36"</td>
                  
                  <td>195</td>
                  
                </tr>
                
                <tr>
                  
                  <td>Enya</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>236</td>
                  
                  <td>Myers</td>
                  
                  <td>212.32281</td>
                  
                  <td>paid</td>
                  
                  <td>Grand Rapids-Wyoming, MI</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1529995579000.0</td>
                  
                  <td>235</td>
                  
                  <td>May It Be (Album version)</td>
                  
                  <td>200</td>
                  
                  <td>1538436387000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>236</td>
                  
                </tr>
                
                <tr>
                  
                  <td>M.I.A.</td>
                  
                  <td>Logged In</td>
                  
                  <td>Alex</td>
                  
                  <td>M</td>
                  
                  <td>48</td>
                  
                  <td>Hogan</td>
                  
                  <td>206.13179</td>
                  
                  <td>free</td>
                  
                  <td>Denver-Aurora-Lakewood, CO</td>
                  
                  <td>PUT</td>
                  
                  <td>NextSong</td>
                  
                  <td>1535066380000.0</td>
                  
                  <td>523</td>
                  
                  <td>Paper Planes</td>
                  
                  <td>200</td>
                  
                  <td>1538437394000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.2; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>101</td>
                  
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
      var tableWrapper = $('.df-table-wrapper-28f0c736');
      var fixedHeader = $('.fixed-header', tableWrapper);
      var tableContainer = $('.df-table-container', tableWrapper);
      var table = $('.df-table', tableContainer);
      var rows = $('tbody > tr', table);
      var total = 100;
  
      fixedHeader
        .css('width', table.width())
        .find('.fixed-cell')
        .each(function(i, e) {
          $(this).css('width', $('.df-table-wrapper-28f0c736 th:nth-child(' + (i+1) + ')').css('width'));
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
    <div id="chartFigure9c339a58" class="pd_save is-viewer-good" style="overflow-x:auto">
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
  
  
  <div class="df-table-wrapper df-table-wrapper-9c339a58 panel-group pd_save">
    <!-- dataframe schema -->
    
    <div class="panel panel-default">
      <div class="panel-heading">
        <h4 class="panel-title" style="margin: 0px;">
          <a data-toggle="collapse" href="#df-schema-9c339a58" data-parent="#df-table-wrapper-9c339a58">Schema</a>
        </h4>
      </div>
      <div id="df-schema-9c339a58" class="panel-collapse collapse">
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
            
          </div>
        </div>
      </div>
    </div>
    
    <!-- dataframe table -->
    <div class="panel panel-default">
      
      <div class="panel-heading">
        <h4 class="panel-title" style="margin: 0px;">
          <a data-toggle="collapse" href="#df-table-9c339a58" data-parent="#df-table-wrapper-9c339a58"> Table</a>
        </h4>
      </div>
      
      <div id="df-table-9c339a58" class="panel-collapse collapse in">
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
                  
                </tr>
              </thead>
              <tbody>
                
                <tr>
                  
                  <td>148</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>28</td>
                  
                  <td>4</td>
                  
                  <td>10</td>
                  
                  <td>3</td>
                  
                  <td>13</td>
                  
                  <td>307</td>
                  
                  <td>69</td>
                  
                  <td>73589</td>
                  
                </tr>
                
                <tr>
                  
                  <td>85</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>31</td>
                  
                  <td>33</td>
                  
                  <td>116</td>
                  
                  <td>39</td>
                  
                  <td>52</td>
                  
                  <td>2223</td>
                  
                  <td>109</td>
                  
                  <td>550730</td>
                  
                </tr>
                
                <tr>
                  
                  <td>251</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>31</td>
                  
                  <td>16</td>
                  
                  <td>117</td>
                  
                  <td>40</td>
                  
                  <td>60</td>
                  
                  <td>2072</td>
                  
                  <td>100</td>
                  
                  <td>506920</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>255</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>21</td>
                  
                  <td>90</td>
                  
                  <td>42</td>
                  
                  <td>51</td>
                  
                  <td>2147</td>
                  
                  <td>65</td>
                  
                  <td>530410</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>78</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>19</td>
                  
                  <td>16</td>
                  
                  <td>66</td>
                  
                  <td>22</td>
                  
                  <td>52</td>
                  
                  <td>1266</td>
                  
                  <td>59</td>
                  
                  <td>306659</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>28</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>35</td>
                  
                  <td>7</td>
                  
                  <td>26</td>
                  
                  <td>14</td>
                  
                  <td>17</td>
                  
                  <td>692</td>
                  
                  <td>27</td>
                  
                  <td>170158</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>6</td>
                  
                  <td>23</td>
                  
                  <td>6</td>
                  
                  <td>13</td>
                  
                  <td>381</td>
                  
                  <td>115</td>
                  
                  <td>92515</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100020</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>14</td>
                  
                  <td>2</td>
                  
                  <td>4</td>
                  
                  <td>1</td>
                  
                  <td>3</td>
                  
                  <td>96</td>
                  
                  <td>89</td>
                  
                  <td>45539</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>200028</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>23</td>
                  
                  <td>28</td>
                  
                  <td>39</td>
                  
                  <td>25</td>
                  
                  <td>32</td>
                  
                  <td>943</td>
                  
                  <td>49</td>
                  
                  <td>231751</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>200019</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>33</td>
                  
                  <td>41</td>
                  
                  <td>67</td>
                  
                  <td>14</td>
                  
                  <td>31</td>
                  
                  <td>1202</td>
                  
                  <td>65</td>
                  
                  <td>296997</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100016</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>6</td>
                  
                  <td>0</td>
                  
                  <td>8</td>
                  
                  <td>2</td>
                  
                  <td>6</td>
                  
                  <td>163</td>
                  
                  <td>40</td>
                  
                  <td>40908</td>
                  
                </tr>
                
                <tr>
                  
                  <td>91</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>43</td>
                  
                  <td>2</td>
                  
                  <td>21</td>
                  
                  <td>10</td>
                  
                  <td>5</td>
                  
                  <td>417</td>
                  
                  <td>116</td>
                  
                  <td>99994</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>232</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>100039</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>13</td>
                  
                  <td>2</td>
                  
                  <td>7</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>107</td>
                  
                  <td>107</td>
                  
                  <td>24764</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200004</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>200022</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>26</td>
                  
                  <td>8</td>
                  
                  <td>17</td>
                  
                  <td>5</td>
                  
                  <td>10</td>
                  
                  <td>356</td>
                  
                  <td>88</td>
                  
                  <td>90235</td>
                  
                </tr>
                
                <tr>
                  
                  <td>140</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>80</td>
                  
                  <td>59</td>
                  
                  <td>319</td>
                  
                  <td>125</td>
                  
                  <td>165</td>
                  
                  <td>6233</td>
                  
                  <td>80</td>
                  
                  <td>1537605</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100050</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>3</td>
                  
                  <td>16</td>
                  
                  <td>10</td>
                  
                  <td>9</td>
                  
                  <td>354</td>
                  
                  <td>86</td>
                  
                  <td>83219</td>
                  
                </tr>
                
                <tr>
                  
                  <td>52</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>52</td>
                  
                  <td>7</td>
                  
                  <td>44</td>
                  
                  <td>17</td>
                  
                  <td>24</td>
                  
                  <td>888</td>
                  
                  <td>103</td>
                  
                  <td>254295</td>
                  
                </tr>
                
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
                  
                </tr>
                
                <tr>
                  
                  <td>13</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>52</td>
                  
                  <td>4</td>
                  
                  <td>21</td>
                  
                  <td>11</td>
                  
                  <td>10</td>
                  
                  <td>439</td>
                  
                  <td>76</td>
                  
                  <td>110289</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>16</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>21</td>
                  
                  <td>7</td>
                  
                  <td>40</td>
                  
                  <td>10</td>
                  
                  <td>17</td>
                  
                  <td>671</td>
                  
                  <td>53</td>
                  
                  <td>166008</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>100048</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>8</td>
                  
                  <td>1</td>
                  
                  <td>9</td>
                  
                  <td>165</td>
                  
                  <td>89</td>
                  
                  <td>42165</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>142</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>27</td>
                  
                  <td>31</td>
                  
                  <td>104</td>
                  
                  <td>34</td>
                  
                  <td>48</td>
                  
                  <td>1916</td>
                  
                  <td>63</td>
                  
                  <td>481158</td>
                  
                </tr>
                
                <tr>
                  
                  <td>20</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>23</td>
                  
                  <td>115</td>
                  
                  <td>42</td>
                  
                  <td>62</td>
                  
                  <td>2158</td>
                  
                  <td>76</td>
                  
                  <td>534216</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>283</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>35</td>
                  
                  <td>193</td>
                  
                  <td>53</td>
                  
                  <td>108</td>
                  
                  <td>3487</td>
                  
                  <td>61</td>
                  
                  <td>848106</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100027</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>20</td>
                  
                  <td>5</td>
                  
                  <td>18</td>
                  
                  <td>29</td>
                  
                  <td>12</td>
                  
                  <td>483</td>
                  
                  <td>96</td>
                  
                  <td>123839</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>96</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>20</td>
                  
                  <td>78</td>
                  
                  <td>36</td>
                  
                  <td>60</td>
                  
                  <td>1950</td>
                  
                  <td>70</td>
                  
                  <td>483130</td>
                  
                </tr>
                
                <tr>
                  
                  <td>235</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>21</td>
                  
                  <td>3</td>
                  
                  <td>26</td>
                  
                  <td>10</td>
                  
                  <td>15</td>
                  
                  <td>539</td>
                  
                  <td>77</td>
                  
                  <td>133705</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100031</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>37</td>
                  
                  <td>2</td>
                  
                  <td>9</td>
                  
                  <td>6</td>
                  
                  <td>3</td>
                  
                  <td>249</td>
                  
                  <td>182</td>
                  
                  <td>59388</td>
                  
                </tr>
                
                <tr>
                  
                  <td>268</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>13</td>
                  
                  <td>7</td>
                  
                  <td>44</td>
                  
                  <td>16</td>
                  
                  <td>22</td>
                  
                  <td>835</td>
                  
                  <td>63</td>
                  
                  <td>215666</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>300046</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>6</td>
                  
                  <td>8</td>
                  
                  <td>50</td>
                  
                  <td>14</td>
                  
                  <td>23</td>
                  
                  <td>636</td>
                  
                  <td>76</td>
                  
                  <td>155876</td>
                  
                </tr>
                
                <tr>
                  
                  <td>262</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>11</td>
                  
                  <td>0</td>
                  
                  <td>7</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>117</td>
                  
                  <td>90</td>
                  
                  <td>26711</td>
                  
                </tr>
                
                <tr>
                  
                  <td>269</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>30</td>
                  
                  <td>36</td>
                  
                  <td>161</td>
                  
                  <td>71</td>
                  
                  <td>73</td>
                  
                  <td>2900</td>
                  
                  <td>89</td>
                  
                  <td>750481</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>17</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>48</td>
                  
                  <td>21</td>
                  
                  <td>84</td>
                  
                  <td>15</td>
                  
                  <td>67</td>
                  
                  <td>1693</td>
                  
                  <td>61</td>
                  
                  <td>464182</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>175</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>83</td>
                  
                  <td>13</td>
                  
                  <td>112</td>
                  
                  <td>39</td>
                  
                  <td>55</td>
                  
                  <td>2049</td>
                  
                  <td>69</td>
                  
                  <td>507035</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300049</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>41</td>
                  
                  <td>70</td>
                  
                  <td>506</td>
                  
                  <td>140</td>
                  
                  <td>191</td>
                  
                  <td>5879</td>
                  
                  <td>70</td>
                  
                  <td>1452559</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100046</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>40</td>
                  
                  <td>5</td>
                  
                  <td>24</td>
                  
                  <td>2</td>
                  
                  <td>8</td>
                  
                  <td>487</td>
                  
                  <td>103</td>
                  
                  <td>120422</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>8</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>25</td>
                  
                  <td>82</td>
                  
                  <td>5800</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>100013</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>20</td>
                  
                  <td>13</td>
                  
                  <td>4525</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200020</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>3</td>
                  
                  <td>88</td>
                  
                  <td>66</td>
                  
                  <td>109</td>
                  
                  <td>35</td>
                  
                  <td>54</td>
                  
                  <td>2112</td>
                  
                  <td>79</td>
                  
                  <td>514153</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>171</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>23</td>
                  
                  <td>12</td>
                  
                  <td>19</td>
                  
                  <td>695</td>
                  
                  <td>61</td>
                  
                  <td>175567</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>216</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>288</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>18</td>
                  
                  <td>13</td>
                  
                  <td>77</td>
                  
                  <td>27</td>
                  
                  <td>37</td>
                  
                  <td>1334</td>
                  
                  <td>226</td>
                  
                  <td>328117</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>200024</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>35</td>
                  
                  <td>53</td>
                  
                  <td>14</td>
                  
                  <td>40</td>
                  
                  <td>960</td>
                  
                  <td>50</td>
                  
                  <td>240375</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>143</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>16</td>
                  
                  <td>22</td>
                  
                  <td>100</td>
                  
                  <td>45</td>
                  
                  <td>55</td>
                  
                  <td>2081</td>
                  
                  <td>108</td>
                  
                  <td>510014</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100051</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>13</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>69</td>
                  
                  <td>0</td>
                  
                  <td>16450</td>
                  
                </tr>
                
                <tr>
                  
                  <td>32</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>7</td>
                  
                  <td>1</td>
                  
                  <td>9</td>
                  
                  <td>1</td>
                  
                  <td>11</td>
                  
                  <td>201</td>
                  
                  <td>31</td>
                  
                  <td>48420</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>100045</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>65</td>
                  
                  <td>27</td>
                  
                  <td>16280</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>141</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>21</td>
                  
                  <td>0</td>
                  
                  <td>16</td>
                  
                  <td>4</td>
                  
                  <td>11</td>
                  
                  <td>249</td>
                  
                  <td>88</td>
                  
                  <td>61174</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>56</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>6</td>
                  
                  <td>14</td>
                  
                  <td>81</td>
                  
                  <td>29</td>
                  
                  <td>65</td>
                  
                  <td>1679</td>
                  
                  <td>61</td>
                  
                  <td>416337</td>
                  
                </tr>
                
                <tr>
                  
                  <td>213</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>49</td>
                  
                  <td>17</td>
                  
                  <td>28</td>
                  
                  <td>861</td>
                  
                  <td>59</td>
                  
                  <td>232798</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100017</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>24</td>
                  
                  <td>3</td>
                  
                  <td>9</td>
                  
                  <td>3</td>
                  
                  <td>1</td>
                  
                  <td>127</td>
                  
                  <td>103</td>
                  
                  <td>32449</td>
                  
                </tr>
                
                <tr>
                  
                  <td>150</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>10</td>
                  
                  <td>1</td>
                  
                  <td>10</td>
                  
                  <td>2</td>
                  
                  <td>4</td>
                  
                  <td>155</td>
                  
                  <td>123</td>
                  
                  <td>44164</td>
                  
                </tr>
                
                <tr>
                  
                  <td>106</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>17</td>
                  
                  <td>15</td>
                  
                  <td>86</td>
                  
                  <td>38</td>
                  
                  <td>44</td>
                  
                  <td>1649</td>
                  
                  <td>64</td>
                  
                  <td>415610</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>270</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>10</td>
                  
                  <td>9</td>
                  
                  <td>48</td>
                  
                  <td>11</td>
                  
                  <td>25</td>
                  
                  <td>765</td>
                  
                  <td>78</td>
                  
                  <td>192780</td>
                  
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
                  
                </tr>
                
                <tr>
                  
                  <td>100026</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>13</td>
                  
                  <td>1</td>
                  
                  <td>22</td>
                  
                  <td>9</td>
                  
                  <td>9</td>
                  
                  <td>406</td>
                  
                  <td>232</td>
                  
                  <td>96001</td>
                  
                </tr>
                
                <tr>
                  
                  <td>30</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>37</td>
                  
                  <td>37</td>
                  
                  <td>177</td>
                  
                  <td>69</td>
                  
                  <td>103</td>
                  
                  <td>3988</td>
                  
                  <td>63</td>
                  
                  <td>984086</td>
                  
                </tr>
                
                <tr>
                  
                  <td>66</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>15</td>
                  
                  <td>87</td>
                  
                  <td>24</td>
                  
                  <td>43</td>
                  
                  <td>1804</td>
                  
                  <td>126</td>
                  
                  <td>448107</td>
                  
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
      var tableWrapper = $('.df-table-wrapper-9c339a58');
      var fixedHeader = $('.fixed-header', tableWrapper);
      var tableContainer = $('.df-table-container', tableWrapper);
      var table = $('.df-table', tableContainer);
      var rows = $('tbody > tr', table);
      var total = 100;
  
      fixedHeader
        .css('width', table.width())
        .find('.fixed-cell')
        .each(function(i, e) {
          $(this).css('width', $('.df-table-wrapper-9c339a58 th:nth-child(' + (i+1) + ')').css('width'));
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

    Explained variance by 3 principal components: 88.07%


# Modeling

## Split in training, test, validation set


```python
train, test = features_df.randomSplit([0.8, 0.2], seed=42)

plt.hist(features_df.toPandas()['label'])
plt.show()
```


![png](output_52_0.png)


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

# create evaluator
evaluator = BinaryClassificationEvaluator(metricName = 'areaUnderPR')

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

## Model Evaluation

* use scikit learn metrics f1, precision, recall for model evaluation


```python
# evaluate results
pd_pred = pred.toPandas()

# show resulting format
pd_pred.head(3)

# calculate score for f1, precision, recall
f1 = f1_score(pd_pred.label, pd_pred.prediction)
recall = recall_score(pd_pred.label, pd_pred.prediction)
precision = precision_score(pd_pred.label, pd_pred.prediction)

print("F1 Score: {:.2f}, Recall: {:.2f}, Precision: {:.2f}".format(f1, recall, precision))
```

    F1 Score: 0.23, Recall: 0.33, Precision: 0.18

