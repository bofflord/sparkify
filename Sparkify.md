
# Sparkify Project Workspace
This workspace contains a tiny subset (128MB) of the full dataset available (12GB). Feel free to use this workspace to build your project, or to explore a smaller subset with Spark before deploying your cluster on the cloud. Instructions for setting up your Spark cluster is included in the last lesson of the Extracurricular Spark Course content.

You can follow the steps below to guide your data analysis and model building portion of this project.


```python
# import libraries

import numpy as np
import pandas as pd
import seaborn as sns

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
    Requirement already satisfied: astunparse in /opt/conda/lib/python3.6/site-packages (from pixiedust) (1.6.2)
    Requirement already satisfied: markdown in /opt/conda/lib/python3.6/site-packages (from pixiedust) (2.6.9)
    Requirement already satisfied: colour in /opt/conda/lib/python3.6/site-packages (from pixiedust) (0.1.5)
    Requirement already satisfied: geojson in /opt/conda/lib/python3.6/site-packages (from pixiedust) (2.5.0)
    Requirement already satisfied: mpld3 in /opt/conda/lib/python3.6/site-packages (from pixiedust) (0.3)
    Requirement already satisfied: requests in /opt/conda/lib/python3.6/site-packages (from pixiedust) (2.18.4)
    Requirement already satisfied: lxml in /opt/conda/lib/python3.6/site-packages (from pixiedust) (4.1.1)
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

# Exploratory Data Analysis
When you're working with the full dataset, perform EDA by loading a small subset of the data and doing basic manipulations within Spark. In this workspace, you are already provided a small subset of data you can explore.

### Define Churn

Once you've done some preliminary analysis, create a column `Churn` to use as the label for your model. I suggest using the `Cancellation Confirmation` events to define your churn, which happen for both paid and free users. As a bonus task, you can also look into the `Downgrade` events.

### Explore Data
Once you've defined churn, perform some exploratory data analysis to observe the behavior for users who stayed vs users who churned. You can start by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.

#### Exploratory Data Analysis


```python
display(user_log)
```


<style type="text/css">.pd_warning{display:none;}</style><div class="pd_warning"><em>Hey, there's something awesome here! To see it, open this notebook outside GitHub, in a viewer like Jupyter</em></div>
        <div class="pd_save is-viewer-good" style="padding-right:10px;text-align: center;line-height:initial !important;font-size: xx-large;font-weight: 500;color: coral;">
            
        </div>
    <div id="chartFigure759a21e4" class="pd_save is-viewer-good" style="overflow-x:auto">
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
  
  
  <div class="df-table-wrapper df-table-wrapper-759a21e4 panel-group pd_save">
    <!-- dataframe schema -->
    
    <div class="panel panel-default">
      <div class="panel-heading">
        <h4 class="panel-title" style="margin: 0px;">
          <a data-toggle="collapse" href="#df-schema-759a21e4" data-parent="#df-table-wrapper-759a21e4">Schema</a>
        </h4>
      </div>
      <div id="df-schema-759a21e4" class="panel-collapse collapse">
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
            
              <div class="df-schema-field"><strong>churn: </strong> int32</div>
            
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
          <a data-toggle="collapse" href="#df-table-759a21e4" data-parent="#df-table-wrapper-759a21e4"> Table</a>
        </h4>
      </div>
      
      <div id="df-table-759a21e4" class="panel-collapse collapse in">
        <div class="panel-body">
          
          <input type="text" class="df-table-search form-control input-sm" placeholder="Search table">
          
          <div>
            
            <span class="df-table-search-count">Showing 21 of 278154 rows</span>
            
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
                
                <div class="fixed-cell">churn</div>
                
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
                  
                  <th><div>churn</div></th>
                  
                  <th><div>gender_bin</div></th>
                  
                  <th><div>level_bin</div></th>
                  
                </tr>
              </thead>
              <tbody>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Payton</td>
                  
                  <td>F</td>
                  
                  <td>30</td>
                  
                  <td>Campbell</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>GET</td>
                  
                  <td>About</td>
                  
                  <td>1529027541000</td>
                  
                  <td>267</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538380294000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"</td>
                  
                  <td>39</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 07:51:34</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Faigy</td>
                  
                  <td>F</td>
                  
                  <td>97</td>
                  
                  <td>Howe</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1538211832000</td>
                  
                  <td>248</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538388298000</td>
                  
                  <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>95</td>
                  
                  <td>AZ</td>
                  
                  <td>2018-10-01 10:04:58</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Grant</td>
                  
                  <td>M</td>
                  
                  <td>102</td>
                  
                  <td>Flores</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>Logout</td>
                  
                  <td>1538120859000</td>
                  
                  <td>141</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538391799000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>142</td>
                  
                  <td>NY-NJ-PA</td>
                  
                  <td>2018-10-01 11:03:19</td>
                  
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
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Grant</td>
                  
                  <td>M</td>
                  
                  <td>110</td>
                  
                  <td>Flores</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>New York-Newark-Jersey City, NY-NJ-PA</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1538120859000</td>
                  
                  <td>141</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538393141000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>142</td>
                  
                  <td>NY-NJ-PA</td>
                  
                  <td>2018-10-01 11:25:41</td>
                  
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
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>6</td>
                  
                  <td>Raymond</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Hartford-West Hartford-East Hartford, CT</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Down</td>
                  
                  <td>1534245996000</td>
                  
                  <td>26</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538401005000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>27</td>
                  
                  <td>CT</td>
                  
                  <td>2018-10-01 13:36:45</td>
                  
                  <td>13</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
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
                  
                  <td>Panav</td>
                  
                  <td>M</td>
                  
                  <td>62</td>
                  
                  <td>Myers</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Morgantown, WV</td>
                  
                  <td>PUT</td>
                  
                  <td>Add to Playlist</td>
                  
                  <td>1538239045000</td>
                  
                  <td>237</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538401601000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>15</td>
                  
                  <td>WV</td>
                  
                  <td>2018-10-01 13:46:41</td>
                  
                  <td>13</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Giovanni</td>
                  
                  <td>M</td>
                  
                  <td>74</td>
                  
                  <td>Garcia</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Myrtle Beach-Conway-North Myrtle Beach, SC-NC</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1536138073000</td>
                  
                  <td>223</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538401665000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>146</td>
                  
                  <td>SC-NC</td>
                  
                  <td>2018-10-01 13:47:45</td>
                  
                  <td>13</td>
                  
                  <td>2018-10-01 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
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
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>50</td>
                  
                  <td>Johnson</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Lexington-Fayette, KY</td>
                  
                  <td>GET</td>
                  
                  <td>Settings</td>
                  
                  <td>1538080987000</td>
                  
                  <td>236</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538406550000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>51</td>
                  
                  <td>KY</td>
                  
                  <td>2018-10-01 15:09:10</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Faigy</td>
                  
                  <td>F</td>
                  
                  <td>184</td>
                  
                  <td>Howe</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Phoenix-Mesa-Scottsdale, AZ</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1538211832000</td>
                  
                  <td>248</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538408219000</td>
                  
                  <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>95</td>
                  
                  <td>AZ</td>
                  
                  <td>2018-10-01 15:36:59</td>
                  
                  <td>15</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Riley</td>
                  
                  <td>F</td>
                  
                  <td>19</td>
                  
                  <td>Taylor</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Boston-Cambridge-Newton, MA-NH</td>
                  
                  <td>PUT</td>
                  
                  <td>Logout</td>
                  
                  <td>1536403972000</td>
                  
                  <td>280</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538414023000</td>
                  
                  <td>"Mozilla/5.0 (iPad; CPU OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>92</td>
                  
                  <td>MA-NH</td>
                  
                  <td>2018-10-01 17:13:43</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Sebastian</td>
                  
                  <td>M</td>
                  
                  <td>4</td>
                  
                  <td>Wang</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Jackson, MS</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1538050164000</td>
                  
                  <td>52</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538419663000</td>
                  
                  <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"</td>
                  
                  <td>53</td>
                  
                  <td>MS</td>
                  
                  <td>2018-10-01 18:47:43</td>
                  
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
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Payton</td>
                  
                  <td>F</td>
                  
                  <td>7</td>
                  
                  <td>Campbell</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1529027541000</td>
                  
                  <td>293</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538420791000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"</td>
                  
                  <td>39</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-01 19:06:31</td>
                  
                  <td>19</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Spencer</td>
                  
                  <td>M</td>
                  
                  <td>40</td>
                  
                  <td>Gonzalez</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Concord, NH</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1537347211000</td>
                  
                  <td>64</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538424906000</td>
                  
                  <td>Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>65</td>
                  
                  <td>NH</td>
                  
                  <td>2018-10-01 20:15:06</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Ethan</td>
                  
                  <td>M</td>
                  
                  <td>266</td>
                  
                  <td>Johnson</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Lexington-Fayette, KY</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1538080987000</td>
                  
                  <td>236</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538451642000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>51</td>
                  
                  <td>KY</td>
                  
                  <td>2018-10-02 03:40:42</td>
                  
                  <td>3</td>
                  
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
                  
                  <td>1</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Christian</td>
                  
                  <td>M</td>
                  
                  <td>2</td>
                  
                  <td>Robinson</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Quincy, IL-MO</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1534942082000</td>
                  
                  <td>300</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538460954000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>87</td>
                  
                  <td>IL-MO</td>
                  
                  <td>2018-10-02 06:15:54</td>
                  
                  <td>6</td>
                  
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
                  
                  <td>0</td>
                  
                </tr>
                
                <tr>
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Christian</td>
                  
                  <td>M</td>
                  
                  <td>11</td>
                  
                  <td>Robinson</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Quincy, IL-MO</td>
                  
                  <td>PUT</td>
                  
                  <td>Add to Playlist</td>
                  
                  <td>1534942082000</td>
                  
                  <td>300</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538462476000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"</td>
                  
                  <td>87</td>
                  
                  <td>IL-MO</td>
                  
                  <td>2018-10-02 06:41:16</td>
                  
                  <td>6</td>
                  
                  <td>2018-10-02 00:00:00</td>
                  
                  <td>0</td>
                  
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
                  
                  <td>None</td>
                  
                  <td>Logged In</td>
                  
                  <td>Payton</td>
                  
                  <td>F</td>
                  
                  <td>2</td>
                  
                  <td>Campbell</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Los Angeles-Long Beach-Anaheim, CA</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1529027541000</td>
                  
                  <td>323</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538463870000</td>
                  
                  <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"</td>
                  
                  <td>39</td>
                  
                  <td>CA</td>
                  
                  <td>2018-10-02 07:04:30</td>
                  
                  <td>7</td>
                  
                  <td>2018-10-02 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
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
                  
                  <td>Riley</td>
                  
                  <td>F</td>
                  
                  <td>45</td>
                  
                  <td>Taylor</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Boston-Cambridge-Newton, MA-NH</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1536403972000</td>
                  
                  <td>322</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538493206000</td>
                  
                  <td>"Mozilla/5.0 (iPad; CPU OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>92</td>
                  
                  <td>MA-NH</td>
                  
                  <td>2018-10-02 15:13:26</td>
                  
                  <td>15</td>
                  
                  <td>2018-10-02 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
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
                  
                  <td>Colin</td>
                  
                  <td>M</td>
                  
                  <td>33</td>
                  
                  <td>Larson</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Dallas-Fort Worth-Arlington, TX</td>
                  
                  <td>GET</td>
                  
                  <td>Roll Advert</td>
                  
                  <td>1537982255000</td>
                  
                  <td>256</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538499115000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0</td>
                  
                  <td>100</td>
                  
                  <td>TX</td>
                  
                  <td>2018-10-02 16:51:55</td>
                  
                  <td>16</td>
                  
                  <td>2018-10-02 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
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
                  
                  <td>Riley</td>
                  
                  <td>F</td>
                  
                  <td>92</td>
                  
                  <td>Taylor</td>
                  
                  <td>nan</td>
                  
                  <td>free</td>
                  
                  <td>Boston-Cambridge-Newton, MA-NH</td>
                  
                  <td>PUT</td>
                  
                  <td>Thumbs Up</td>
                  
                  <td>1536403972000</td>
                  
                  <td>322</td>
                  
                  <td>None</td>
                  
                  <td>307</td>
                  
                  <td>1538501591000</td>
                  
                  <td>"Mozilla/5.0 (iPad; CPU OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"</td>
                  
                  <td>92</td>
                  
                  <td>MA-NH</td>
                  
                  <td>2018-10-02 17:33:11</td>
                  
                  <td>17</td>
                  
                  <td>2018-10-02 00:00:00</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
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
                  
                  <td>19</td>
                  
                  <td>Hogan</td>
                  
                  <td>nan</td>
                  
                  <td>paid</td>
                  
                  <td>Denver-Aurora-Lakewood, CO</td>
                  
                  <td>GET</td>
                  
                  <td>Home</td>
                  
                  <td>1535066380000</td>
                  
                  <td>312</td>
                  
                  <td>None</td>
                  
                  <td>200</td>
                  
                  <td>1538502750000</td>
                  
                  <td>Mozilla/5.0 (Windows NT 6.2; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0</td>
                  
                  <td>101</td>
                  
                  <td>CO</td>
                  
                  <td>2018-10-02 17:52:30</td>
                  
                  <td>17</td>
                  
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
      var tableWrapper = $('.df-table-wrapper-759a21e4');
      var fixedHeader = $('.fixed-header', tableWrapper);
      var tableContainer = $('.df-table-container', tableWrapper);
      var table = $('.df-table', tableContainer);
      var rows = $('tbody > tr', table);
      var total = 21;
  
      fixedHeader
        .css('width', table.width())
        .find('.fixed-cell')
        .each(function(i, e) {
          $(this).css('width', $('.df-table-wrapper-759a21e4 th:nth-child(' + (i+1) + ')').css('width'));
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
    column artist : 8249
    column firstName : 8249
    column gender : 8249
    column lastName : 8249
    column length : 8249
    column location : 8249
    column registration : 8249
    column song : 8249
    column userAgent : 8249
    column userId : 0
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
    column userId : 0
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
    column userId : 0
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
    column userId : 0


### Remove values where "auth" value is either "Guest" or "Logged Out" since userId is missing


```python
# remove values where "auth" value is either "Guest" or "Logged Out" since userId is missing
user_log = user_log.filter(user_log["auth"].isin(*["Guest", "Logged Out"]) == False)
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

#### (4) "location": analyse user locations and CSA (Combined Statistical Areas)


```python
# create new feature CSA (Combined Statistical Areas) from location 
get_csa = F.split(user_log["location"], ", ")
user_log = user_log.withColumn("CSA", get_csa.getItem(1))
```


```python
distinct_users_csa = user_log.select("userId", "CSA").distinct().count()
distinct_users_location = user_log.select("userId", "location").distinct().count()
print("Distinct users per CSA: {}. Distinct users per location: {}".format(distinct_users_csa, distinct_users_location))
```

    Distinct users per CSA: 225. Distinct users per location: 225



```python
# number of users per CSA
pd_distinct_users_csa = user_log.select("userId", "CSA").distinct().groupBy("CSA").agg(F.count("userId").alias("userId_count")).orderBy("userId_count", ascending=False).toPandas()
```


```python
plt.figure(figsize=(16, 16))
sns.barplot(x="userId_count", y="CSA", data=pd_distinct_users_csa);
```


![png](output_27_0.png)


####  (5) convert to time/date: ####
list = ["registration", "ts"]


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



![png](output_35_1.png)


## Features from "page" value ##

### Selection of page values for new features:
* "downgraded" from "Submit Downgrad"
* ...


```python
def create_page_value_feature(df, page_value, col_name):
    '''
    ARGS:
    OUTPUT
    
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

## Define feature "churn"


```python
flag_churn_event = F.udf(lambda page: 1 if page == "Cancel" else 0, IntegerType())
user_log = user_log.withColumn("churn", flag_churn_event("page"))
```

# Feature Engineering
Once you've familiarized yourself with the data, build out the features you find promising to train your model on. To work with the full dataset, you can follow the following steps.
- Write a script to extract the necessary features from the smaller subset of data
- Ensure that your script is scalable, using the best practices discussed in Lesson 3
- Try your script on the full data set, debugging your script if necessary

If you are working in the classroom workspace, you can just extract features based on the small subset of data contained here. Be sure to transfer over this work to the larger dataset when you work on your Spark cluster.

## Create new dataframe for features bases on userId's


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


```python
page_features_count = user_log.groupBy("userId").sum('downgraded', 'upgraded',
 'advert_shown',
 'thumps_down',
 'thumps_up',
 'friend_added',
 'song_added')
features_df = features_df.join(page_features_count, "userId", how="left")
```

## Encode further features
* "song_count": songs per user
* "days_since_reg": days from registration until latest user timestamp of a user


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

## Explore data with regards to churn


```python
display(features_df)
```


<style type="text/css">.pd_warning{display:none;}</style><div class="pd_warning"><em>Hey, there's something awesome here! To see it, open this notebook outside GitHub, in a viewer like Jupyter</em></div>
        <div class="pd_save is-viewer-good" style="padding-right:10px;text-align: center;line-height:initial !important;font-size: xx-large;font-weight: 500;color: coral;">
            
        </div>
    <div id="chartFigurece5ca45d" class="pd_save is-viewer-good" style="overflow-x:auto">
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
  
  
  <div class="df-table-wrapper df-table-wrapper-ce5ca45d panel-group pd_save">
    <!-- dataframe schema -->
    
    <div class="panel panel-default">
      <div class="panel-heading">
        <h4 class="panel-title" style="margin: 0px;">
          <a data-toggle="collapse" href="#df-schema-ce5ca45d" data-parent="#df-table-wrapper-ce5ca45d">Schema</a>
        </h4>
      </div>
      <div id="df-schema-ce5ca45d" class="panel-collapse collapse">
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
          <a data-toggle="collapse" href="#df-table-ce5ca45d" data-parent="#df-table-wrapper-ce5ca45d"> Table</a>
        </h4>
      </div>
      
      <div id="df-table-ce5ca45d" class="panel-collapse collapse in">
        <div class="panel-body">
          
          <input type="text" class="df-table-search form-control input-sm" placeholder="Search table">
          
          <div>
            
            <span class="df-table-search-count">Showing 100 of 225 rows</span>
            
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
                  
                  <td>4</td>
                  
                  <td>154</td>
                  
                  <td>124</td>
                  
                  <td>38027</td>
                  
                </tr>
                
                <tr>
                  
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
                  
                  <td>68</td>
                  
                  <td>2113</td>
                  
                  <td>71</td>
                  
                  <td>530161</td>
                  
                </tr>
                
                <tr>
                  
                  <td>53</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>25</td>
                  
                  <td>16</td>
                  
                  <td>69</td>
                  
                  <td>25</td>
                  
                  <td>46</td>
                  
                  <td>1746</td>
                  
                  <td>53</td>
                  
                  <td>424898</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100003</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>9</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>51</td>
                  
                  <td>23</td>
                  
                  <td>11894</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200021</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>11</td>
                  
                  <td>35</td>
                  
                  <td>55</td>
                  
                  <td>19</td>
                  
                  <td>30</td>
                  
                  <td>1227</td>
                  
                  <td>71</td>
                  
                  <td>296928</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300011</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>21</td>
                  
                  <td>41</td>
                  
                  <td>437</td>
                  
                  <td>93</td>
                  
                  <td>146</td>
                  
                  <td>4619</td>
                  
                  <td>62</td>
                  
                  <td>1150533</td>
                  
                </tr>
                
                <tr>
                  
                  <td>34</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>53</td>
                  
                  <td>71</td>
                  
                  <td>12316</td>
                  
                </tr>
                
                <tr>
                  
                  <td>101</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>16</td>
                  
                  <td>86</td>
                  
                  <td>29</td>
                  
                  <td>61</td>
                  
                  <td>1797</td>
                  
                  <td>54</td>
                  
                  <td>490548</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100005</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>18</td>
                  
                  <td>3</td>
                  
                  <td>7</td>
                  
                  <td>3</td>
                  
                  <td>3</td>
                  
                  <td>154</td>
                  
                  <td>85</td>
                  
                  <td>36056</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300010</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>25</td>
                  
                  <td>1</td>
                  
                  <td>21</td>
                  
                  <td>8</td>
                  
                  <td>7</td>
                  
                  <td>263</td>
                  
                  <td>74</td>
                  
                  <td>65280</td>
                  
                </tr>
                
                <tr>
                  
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
                  
                  <td>4</td>
                  
                  <td>212</td>
                  
                  <td>57</td>
                  
                  <td>86597</td>
                  
                </tr>
                
                <tr>
                  
                  <td>26</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>22</td>
                  
                  <td>1</td>
                  
                  <td>12</td>
                  
                  <td>6</td>
                  
                  <td>8</td>
                  
                  <td>255</td>
                  
                  <td>77</td>
                  
                  <td>92401</td>
                  
                </tr>
                
                <tr>
                  
                  <td>12</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>21</td>
                  
                  <td>9</td>
                  
                  <td>42</td>
                  
                  <td>13</td>
                  
                  <td>19</td>
                  
                  <td>867</td>
                  
                  <td>73</td>
                  
                  <td>247073</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300006</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>14</td>
                  
                  <td>3</td>
                  
                  <td>23</td>
                  
                  <td>17</td>
                  
                  <td>7</td>
                  
                  <td>279</td>
                  
                  <td>89</td>
                  
                  <td>65964</td>
                  
                </tr>
                
                <tr>
                  
                  <td>91</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>24</td>
                  
                  <td>124</td>
                  
                  <td>42</td>
                  
                  <td>64</td>
                  
                  <td>2580</td>
                  
                  <td>116</td>
                  
                  <td>637146</td>
                  
                </tr>
                
                <tr>
                  
                  <td>22</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>3</td>
                  
                  <td>0</td>
                  
                  <td>28</td>
                  
                  <td>61</td>
                  
                  <td>6398</td>
                  
                </tr>
                
                <tr>
                  
                  <td>93</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>43</td>
                  
                  <td>4</td>
                  
                  <td>35</td>
                  
                  <td>13</td>
                  
                  <td>17</td>
                  
                  <td>640</td>
                  
                  <td>71</td>
                  
                  <td>155952</td>
                  
                </tr>
                
                <tr>
                  
                  <td>111</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>3</td>
                  
                  <td>6</td>
                  
                  <td>34</td>
                  
                  <td>23</td>
                  
                  <td>21</td>
                  
                  <td>698</td>
                  
                  <td>82</td>
                  
                  <td>194266</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300024</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>6</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>88</td>
                  
                  <td>39</td>
                  
                  <td>22889</td>
                  
                </tr>
                
                <tr>
                  
                  <td>140</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>4</td>
                  
                  <td>87</td>
                  
                  <td>75</td>
                  
                  <td>277</td>
                  
                  <td>143</td>
                  
                  <td>148</td>
                  
                  <td>5664</td>
                  
                  <td>80</td>
                  
                  <td>1397558</td>
                  
                </tr>
                
                <tr>
                  
                  <td>13</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>74</td>
                  
                  <td>14</td>
                  
                  <td>57</td>
                  
                  <td>32</td>
                  
                  <td>37</td>
                  
                  <td>1280</td>
                  
                  <td>119</td>
                  
                  <td>306123</td>
                  
                </tr>
                
                <tr>
                  
                  <td>16</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>12</td>
                  
                  <td>4</td>
                  
                  <td>42</td>
                  
                  <td>10</td>
                  
                  <td>19</td>
                  
                  <td>675</td>
                  
                  <td>81</td>
                  
                  <td>168863</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100021</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>30</td>
                  
                  <td>5</td>
                  
                  <td>11</td>
                  
                  <td>7</td>
                  
                  <td>7</td>
                  
                  <td>230</td>
                  
                  <td>65</td>
                  
                  <td>64568</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200003</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>1</td>
                  
                  <td>54</td>
                  
                  <td>25</td>
                  
                  <td>37</td>
                  
                  <td>14</td>
                  
                  <td>16</td>
                  
                  <td>751</td>
                  
                  <td>153</td>
                  
                  <td>181814</td>
                  
                </tr>
                
                <tr>
                  
                  <td>20</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>20</td>
                  
                  <td>21</td>
                  
                  <td>106</td>
                  
                  <td>25</td>
                  
                  <td>54</td>
                  
                  <td>1807</td>
                  
                  <td>77</td>
                  
                  <td>447916</td>
                  
                </tr>
                
                <tr>
                  
                  <td>57</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>6</td>
                  
                  <td>2</td>
                  
                  <td>3</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>92</td>
                  
                  <td>88</td>
                  
                  <td>20712</td>
                  
                </tr>
                
                <tr>
                  
                  <td>120</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>13</td>
                  
                  <td>81</td>
                  
                  <td>22</td>
                  
                  <td>44</td>
                  
                  <td>1571</td>
                  
                  <td>129</td>
                  
                  <td>389799</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100011</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>11</td>
                  
                  <td>5</td>
                  
                  <td>2663</td>
                  
                </tr>
                
                <tr>
                  
                  <td>19</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>5</td>
                  
                  <td>4</td>
                  
                  <td>8</td>
                  
                  <td>216</td>
                  
                  <td>22</td>
                  
                  <td>54292</td>
                  
                </tr>
                
                <tr>
                  
                  <td>41</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>10</td>
                  
                  <td>76</td>
                  
                  <td>36</td>
                  
                  <td>61</td>
                  
                  <td>1894</td>
                  
                  <td>110</td>
                  
                  <td>473524</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100014</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>3</td>
                  
                  <td>17</td>
                  
                  <td>6</td>
                  
                  <td>7</td>
                  
                  <td>257</td>
                  
                  <td>85</td>
                  
                  <td>66533</td>
                  
                </tr>
                
                <tr>
                  
                  <td>112</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>21</td>
                  
                  <td>3</td>
                  
                  <td>9</td>
                  
                  <td>7</td>
                  
                  <td>7</td>
                  
                  <td>215</td>
                  
                  <td>87</td>
                  
                  <td>50671</td>
                  
                </tr>
                
                <tr>
                  
                  <td>37</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>4</td>
                  
                  <td>11</td>
                  
                  <td>75</td>
                  
                  <td>33</td>
                  
                  <td>37</td>
                  
                  <td>1412</td>
                  
                  <td>94</td>
                  
                  <td>353530</td>
                  
                </tr>
                
                <tr>
                  
                  <td>88</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>23</td>
                  
                  <td>20</td>
                  
                  <td>122</td>
                  
                  <td>28</td>
                  
                  <td>58</td>
                  
                  <td>2045</td>
                  
                  <td>76</td>
                  
                  <td>514685</td>
                  
                </tr>
                
                <tr>
                  
                  <td>107</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>19</td>
                  
                  <td>2</td>
                  
                  <td>14</td>
                  
                  <td>2</td>
                  
                  <td>10</td>
                  
                  <td>246</td>
                  
                  <td>74</td>
                  
                  <td>61515</td>
                  
                </tr>
                
                <tr>
                  
                  <td>9</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>16</td>
                  
                  <td>32</td>
                  
                  <td>118</td>
                  
                  <td>40</td>
                  
                  <td>77</td>
                  
                  <td>2676</td>
                  
                  <td>61</td>
                  
                  <td>655793</td>
                  
                </tr>
                
                <tr>
                  
                  <td>17</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>4</td>
                  
                  <td>13</td>
                  
                  <td>40</td>
                  
                  <td>12</td>
                  
                  <td>30</td>
                  
                  <td>927</td>
                  
                  <td>13</td>
                  
                  <td>252525</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300013</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>38</td>
                  
                  <td>8</td>
                  
                  <td>7</td>
                  
                  <td>335</td>
                  
                  <td>90</td>
                  
                  <td>83706</td>
                  
                </tr>
                
                <tr>
                  
                  <td>72</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>10</td>
                  
                  <td>1</td>
                  
                  <td>6</td>
                  
                  <td>3</td>
                  
                  <td>3</td>
                  
                  <td>85</td>
                  
                  <td>124</td>
                  
                  <td>20059</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200025</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>37</td>
                  
                  <td>27</td>
                  
                  <td>48</td>
                  
                  <td>24</td>
                  
                  <td>16</td>
                  
                  <td>790</td>
                  
                  <td>117</td>
                  
                  <td>191989</td>
                  
                </tr>
                
                <tr>
                  
                  <td>114</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>9</td>
                  
                  <td>12</td>
                  
                  <td>74</td>
                  
                  <td>13</td>
                  
                  <td>34</td>
                  
                  <td>1292</td>
                  
                  <td>71</td>
                  
                  <td>324432</td>
                  
                </tr>
                
                <tr>
                  
                  <td>4</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>4</td>
                  
                  <td>26</td>
                  
                  <td>95</td>
                  
                  <td>46</td>
                  
                  <td>59</td>
                  
                  <td>2048</td>
                  
                  <td>63</td>
                  
                  <td>500727</td>
                  
                </tr>
                
                <tr>
                  
                  <td>59</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>11</td>
                  
                  <td>9</td>
                  
                  <td>30</td>
                  
                  <td>16</td>
                  
                  <td>14</td>
                  
                  <td>724</td>
                  
                  <td>65</td>
                  
                  <td>192058</td>
                  
                </tr>
                
                <tr>
                  
                  <td>8</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>18</td>
                  
                  <td>3</td>
                  
                  <td>16</td>
                  
                  <td>5</td>
                  
                  <td>6</td>
                  
                  <td>251</td>
                  
                  <td>115</td>
                  
                  <td>61690</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>25</td>
                  
                  <td>27</td>
                  
                  <td>148</td>
                  
                  <td>49</td>
                  
                  <td>61</td>
                  
                  <td>2682</td>
                  
                  <td>65</td>
                  
                  <td>664690</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300008</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>16</td>
                  
                  <td>111</td>
                  
                  <td>23</td>
                  
                  <td>43</td>
                  
                  <td>1393</td>
                  
                  <td>91</td>
                  
                  <td>345519</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100013</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>39</td>
                  
                  <td>15</td>
                  
                  <td>39</td>
                  
                  <td>28</td>
                  
                  <td>31</td>
                  
                  <td>1131</td>
                  
                  <td>44</td>
                  
                  <td>277199</td>
                  
                </tr>
                
                <tr>
                  
                  <td>23</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>19</td>
                  
                  <td>6</td>
                  
                  <td>28</td>
                  
                  <td>15</td>
                  
                  <td>21</td>
                  
                  <td>656</td>
                  
                  <td>136</td>
                  
                  <td>163813</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200023</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>3</td>
                  
                  <td>116</td>
                  
                  <td>73</td>
                  
                  <td>163</td>
                  
                  <td>66</td>
                  
                  <td>73</td>
                  
                  <td>2955</td>
                  
                  <td>67</td>
                  
                  <td>757076</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100002</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>195</td>
                  
                  <td>161</td>
                  
                  <td>48284</td>
                  
                </tr>
                
                <tr>
                  
                  <td>7</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>16</td>
                  
                  <td>1</td>
                  
                  <td>7</td>
                  
                  <td>1</td>
                  
                  <td>5</td>
                  
                  <td>150</td>
                  
                  <td>73</td>
                  
                  <td>36809</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200020</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>69</td>
                  
                  <td>36</td>
                  
                  <td>56</td>
                  
                  <td>10</td>
                  
                  <td>32</td>
                  
                  <td>1169</td>
                  
                  <td>76</td>
                  
                  <td>281054</td>
                  
                </tr>
                
                <tr>
                  
                  <td>136</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>13</td>
                  
                  <td>15</td>
                  
                  <td>110</td>
                  
                  <td>60</td>
                  
                  <td>66</td>
                  
                  <td>2124</td>
                  
                  <td>76</td>
                  
                  <td>573780</td>
                  
                </tr>
                
                <tr>
                  
                  <td>87</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>50</td>
                  
                  <td>5</td>
                  
                  <td>32</td>
                  
                  <td>27</td>
                  
                  <td>19</td>
                  
                  <td>767</td>
                  
                  <td>59</td>
                  
                  <td>188431</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300003</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>27</td>
                  
                  <td>99</td>
                  
                  <td>6291</td>
                  
                </tr>
                
                <tr>
                  
                  <td>51</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>21</td>
                  
                  <td>100</td>
                  
                  <td>28</td>
                  
                  <td>52</td>
                  
                  <td>2111</td>
                  
                  <td>20</td>
                  
                  <td>523414</td>
                  
                </tr>
                
                <tr>
                  
                  <td>69</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>3</td>
                  
                  <td>9</td>
                  
                  <td>72</td>
                  
                  <td>12</td>
                  
                  <td>33</td>
                  
                  <td>1125</td>
                  
                  <td>71</td>
                  
                  <td>284410</td>
                  
                </tr>
                
                <tr>
                  
                  <td>129</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>7</td>
                  
                  <td>7</td>
                  
                  <td>11</td>
                  
                  <td>11</td>
                  
                  <td>331</td>
                  
                  <td>17</td>
                  
                  <td>80504</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200015</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>22</td>
                  
                  <td>10</td>
                  
                  <td>10</td>
                  
                  <td>2</td>
                  
                  <td>13</td>
                  
                  <td>258</td>
                  
                  <td>88</td>
                  
                  <td>59488</td>
                  
                </tr>
                
                <tr>
                  
                  <td>97</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>24</td>
                  
                  <td>12</td>
                  
                  <td>108</td>
                  
                  <td>32</td>
                  
                  <td>61</td>
                  
                  <td>1975</td>
                  
                  <td>87</td>
                  
                  <td>482994</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300025</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>13</td>
                  
                  <td>139</td>
                  
                  <td>17</td>
                  
                  <td>42</td>
                  
                  <td>1297</td>
                  
                  <td>77</td>
                  
                  <td>314197</td>
                  
                </tr>
                
                <tr>
                  
                  <td>10</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>4</td>
                  
                  <td>37</td>
                  
                  <td>12</td>
                  
                  <td>9</td>
                  
                  <td>673</td>
                  
                  <td>52</td>
                  
                  <td>165509</td>
                  
                </tr>
                
                <tr>
                  
                  <td>38</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>21</td>
                  
                  <td>65</td>
                  
                  <td>21</td>
                  
                  <td>30</td>
                  
                  <td>1322</td>
                  
                  <td>75</td>
                  
                  <td>328596</td>
                  
                </tr>
                
                <tr>
                  
                  <td>113</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>12</td>
                  
                  <td>82</td>
                  
                  <td>32</td>
                  
                  <td>50</td>
                  
                  <td>1585</td>
                  
                  <td>123</td>
                  
                  <td>393855</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300009</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>15</td>
                  
                  <td>132</td>
                  
                  <td>21</td>
                  
                  <td>44</td>
                  
                  <td>1427</td>
                  
                  <td>101</td>
                  
                  <td>352776</td>
                  
                </tr>
                
                <tr>
                  
                  <td>70</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>4</td>
                  
                  <td>12</td>
                  
                  <td>90</td>
                  
                  <td>26</td>
                  
                  <td>41</td>
                  
                  <td>1490</td>
                  
                  <td>145</td>
                  
                  <td>366890</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300019</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>8</td>
                  
                  <td>83</td>
                  
                  <td>19</td>
                  
                  <td>24</td>
                  
                  <td>798</td>
                  
                  <td>85</td>
                  
                  <td>198434</td>
                  
                </tr>
                
                <tr>
                  
                  <td>121</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>28</td>
                  
                  <td>8</td>
                  
                  <td>28</td>
                  
                  <td>20</td>
                  
                  <td>20</td>
                  
                  <td>726</td>
                  
                  <td>132</td>
                  
                  <td>175976</td>
                  
                </tr>
                
                <tr>
                  
                  <td>125</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>8</td>
                  
                  <td>72</td>
                  
                  <td>1774</td>
                  
                </tr>
                
                <tr>
                  
                  <td>156</td>
                  
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
                  
                  <td>3</td>
                  
                  <td>0</td>
                  
                  <td>420</td>
                  
                </tr>
                
                <tr>
                  
                  <td>143</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>15</td>
                  
                  <td>0</td>
                  
                  <td>5</td>
                  
                  <td>6</td>
                  
                  <td>2</td>
                  
                  <td>101</td>
                  
                  <td>62</td>
                  
                  <td>24428</td>
                  
                </tr>
                
                <tr>
                  
                  <td>32</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>3</td>
                  
                  <td>1</td>
                  
                  <td>7</td>
                  
                  <td>0</td>
                  
                  <td>7</td>
                  
                  <td>80</td>
                  
                  <td>21</td>
                  
                  <td>19593</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100018</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>2</td>
                  
                  <td>80</td>
                  
                  <td>9</td>
                  
                  <td>46</td>
                  
                  <td>23</td>
                  
                  <td>31</td>
                  
                  <td>1002</td>
                  
                  <td>111</td>
                  
                  <td>243416</td>
                  
                </tr>
                
                <tr>
                  
                  <td>151</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>4</td>
                  
                  <td>3</td>
                  
                  <td>2</td>
                  
                  <td>138</td>
                  
                  <td>42</td>
                  
                  <td>32761</td>
                  
                </tr>
                
                <tr>
                  
                  <td>145</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>8</td>
                  
                  <td>61</td>
                  
                  <td>34</td>
                  
                  <td>27</td>
                  
                  <td>1129</td>
                  
                  <td>102</td>
                  
                  <td>284321</td>
                  
                </tr>
                
                <tr>
                  
                  <td>56</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>19</td>
                  
                  <td>7</td>
                  
                  <td>49</td>
                  
                  <td>21</td>
                  
                  <td>20</td>
                  
                  <td>734</td>
                  
                  <td>67</td>
                  
                  <td>176994</td>
                  
                </tr>
                
                <tr>
                  
                  <td>109</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>7</td>
                  
                  <td>5</td>
                  
                  <td>23</td>
                  
                  <td>12</td>
                  
                  <td>16</td>
                  
                  <td>717</td>
                  
                  <td>88</td>
                  
                  <td>177363</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100009</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>42</td>
                  
                  <td>8</td>
                  
                  <td>23</td>
                  
                  <td>7</td>
                  
                  <td>12</td>
                  
                  <td>518</td>
                  
                  <td>38</td>
                  
                  <td>127177</td>
                  
                </tr>
                
                <tr>
                  
                  <td>300005</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>4</td>
                  
                  <td>35</td>
                  
                  <td>8</td>
                  
                  <td>9</td>
                  
                  <td>312</td>
                  
                  <td>157</td>
                  
                  <td>76230</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100017</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>14</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>52</td>
                  
                  <td>79</td>
                  
                  <td>11956</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200018</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>32</td>
                  
                  <td>11</td>
                  
                  <td>13</td>
                  
                  <td>15</td>
                  
                  <td>11</td>
                  
                  <td>367</td>
                  
                  <td>49</td>
                  
                  <td>91426</td>
                  
                </tr>
                
                <tr>
                  
                  <td>58</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>11</td>
                  
                  <td>20</td>
                  
                  <td>94</td>
                  
                  <td>19</td>
                  
                  <td>50</td>
                  
                  <td>1694</td>
                  
                  <td>26</td>
                  
                  <td>418889</td>
                  
                </tr>
                
                <tr>
                  
                  <td>11</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>39</td>
                  
                  <td>9</td>
                  
                  <td>40</td>
                  
                  <td>6</td>
                  
                  <td>20</td>
                  
                  <td>647</td>
                  
                  <td>125</td>
                  
                  <td>154922</td>
                  
                </tr>
                
                <tr>
                  
                  <td>110</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>17</td>
                  
                  <td>2</td>
                  
                  <td>14</td>
                  
                  <td>7</td>
                  
                  <td>3</td>
                  
                  <td>178</td>
                  
                  <td>68</td>
                  
                  <td>57941</td>
                  
                </tr>
                
                <tr>
                  
                  <td>150</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>17</td>
                  
                  <td>0</td>
                  
                  <td>7</td>
                  
                  <td>1</td>
                  
                  <td>8</td>
                  
                  <td>178</td>
                  
                  <td>124</td>
                  
                  <td>52556</td>
                  
                </tr>
                
                <tr>
                  
                  <td>68</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>0</td>
                  
                  <td>2</td>
                  
                  <td>7</td>
                  
                  <td>0</td>
                  
                  <td>29</td>
                  
                  <td>100</td>
                  
                  <td>7649</td>
                  
                </tr>
                
                <tr>
                  
                  <td>106</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>4</td>
                  
                  <td>6</td>
                  
                  <td>43</td>
                  
                  <td>29</td>
                  
                  <td>28</td>
                  
                  <td>839</td>
                  
                  <td>40</td>
                  
                  <td>207189</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200006</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>68</td>
                  
                  <td>19</td>
                  
                  <td>21</td>
                  
                  <td>8</td>
                  
                  <td>17</td>
                  
                  <td>542</td>
                  
                  <td>74</td>
                  
                  <td>128334</td>
                  
                </tr>
                
                <tr>
                  
                  <td>116</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>3</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>4</td>
                  
                  <td>63</td>
                  
                  <td>94</td>
                  
                  <td>24516</td>
                  
                </tr>
                
                <tr>
                  
                  <td>147</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>11</td>
                  
                  <td>3</td>
                  
                  <td>28</td>
                  
                  <td>15</td>
                  
                  <td>12</td>
                  
                  <td>547</td>
                  
                  <td>69</td>
                  
                  <td>134424</td>
                  
                </tr>
                
                <tr>
                  
                  <td>135</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>6</td>
                  
                  <td>72</td>
                  
                  <td>1347</td>
                  
                </tr>
                
                <tr>
                  
                  <td>2</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>6</td>
                  
                  <td>29</td>
                  
                  <td>20</td>
                  
                  <td>13</td>
                  
                  <td>755</td>
                  
                  <td>69</td>
                  
                  <td>186907</td>
                  
                </tr>
                
                <tr>
                  
                  <td>124</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>41</td>
                  
                  <td>171</td>
                  
                  <td>74</td>
                  
                  <td>118</td>
                  
                  <td>4079</td>
                  
                  <td>131</td>
                  
                  <td>1007450</td>
                  
                </tr>
                
                <tr>
                  
                  <td>99</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>10</td>
                  
                  <td>5</td>
                  
                  <td>27</td>
                  
                  <td>12</td>
                  
                  <td>11</td>
                  
                  <td>505</td>
                  
                  <td>133</td>
                  
                  <td>125728</td>
                  
                </tr>
                
                <tr>
                  
                  <td>200011</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>1</td>
                  
                  <td>27</td>
                  
                  <td>24</td>
                  
                  <td>37</td>
                  
                  <td>13</td>
                  
                  <td>18</td>
                  
                  <td>650</td>
                  
                  <td>110</td>
                  
                  <td>163538</td>
                  
                </tr>
                
                <tr>
                  
                  <td>66</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>35</td>
                  
                  <td>8</td>
                  
                  <td>52</td>
                  
                  <td>24</td>
                  
                  <td>38</td>
                  
                  <td>1045</td>
                  
                  <td>126</td>
                  
                  <td>255081</td>
                  
                </tr>
                
                <tr>
                  
                  <td>46</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>13</td>
                  
                  <td>11</td>
                  
                  <td>44</td>
                  
                  <td>18</td>
                  
                  <td>30</td>
                  
                  <td>918</td>
                  
                  <td>63</td>
                  
                  <td>226828</td>
                  
                </tr>
                
                <tr>
                  
                  <td>67</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>17</td>
                  
                  <td>10</td>
                  
                  <td>54</td>
                  
                  <td>14</td>
                  
                  <td>37</td>
                  
                  <td>1098</td>
                  
                  <td>139</td>
                  
                  <td>270992</td>
                  
                </tr>
                
                <tr>
                  
                  <td>100015</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>2</td>
                  
                  <td>69</td>
                  
                  <td>8</td>
                  
                  <td>35</td>
                  
                  <td>14</td>
                  
                  <td>22</td>
                  
                  <td>800</td>
                  
                  <td>68</td>
                  
                  <td>200588</td>
                  
                </tr>
                
                <tr>
                  
                  <td>153</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>0</td>
                  
                  <td>1</td>
                  
                  <td>0</td>
                  
                  <td>4</td>
                  
                  <td>52</td>
                  
                  <td>16</td>
                  
                  <td>32</td>
                  
                  <td>941</td>
                  
                  <td>27</td>
                  
                  <td>236046</td>
                  
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
      var tableWrapper = $('.df-table-wrapper-ce5ca45d');
      var fixedHeader = $('.fixed-header', tableWrapper);
      var tableContainer = $('.df-table-container', tableWrapper);
      var table = $('.df-table', tableContainer);
      var rows = $('tbody > tr', table);
      var total = 100;
  
      fixedHeader
        .css('width', table.width())
        .find('.fixed-cell')
        .each(function(i, e) {
          $(this).css('width', $('.df-table-wrapper-ce5ca45d th:nth-child(' + (i+1) + ')').css('width'));
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
pd_features_df = features_df.toPandas()
```


```python
features_wo_id_label = pd_features_df.columns.values.tolist()[2:]
first_features_df = pd_features_df[features_wo_id_label[:-6]]
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


    <matplotlib.figure.Figure at 0x7fa8a4d58fd0>



![png](output_64_1.png)



```python
# show relationship between variables for all features
plt.figure(figsize=(25, 25))
sns.pairplot(second_features_df, hue="label");
```


    <matplotlib.figure.Figure at 0x7fa8a75f1e80>



![png](output_65_1.png)



```python
# print correlation between variables
corr = pd_features_df.drop("userId", axis=1).corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True));
```


![png](output_66_0.png)


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
    # apply pipelien on features_df Dataframe
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


```python
display(features_df)
```

## Perform PCA to select relevant features


```python
pca = PCA(k=5, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(features_df.select("features"))

result = model.transform(features_df.select("features")).select("pcaFeatures")
result.show(truncate=False)
```

# Modeling
Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. Evaluate the accuracy of the various models, tuning parameters as necessary. Determine your winning model based on test accuracy and report results on the validation set. Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.

## Split in training, test, validation set


```python
train, test = features_df.randomSplit([0.8, 0.2], seed=42)

plt.hist(features_df.toPandas()['label'])
plt.show()
```

### Analyze label class imbalance - tbd +++++++++++++


```python
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


```python
# Create a logistic regression object
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', weightCol="classWeights")

# create evaluator
evaluator = BinaryClassificationEvaluator(metricName = 'areaUnderPR')

```


```python
# create lr_model
lr_model = lr.fit(train)
training_summary = lr_model.summary
```


```python
# ToDo: evaluate training summary ++++++++++++++++

# TBD
```

## Tune Model
* use cross validation via CrossValidator and paramGrid


```python
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
pd_pred.head()
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


```python
# TODO: change label or create feature label

#correct_results = (results.filter(results.label == results.prediction).count())
#total_results= (results.count())
#accuracy = correct_results/ total_results
```

# Final Steps
Clean up your code, adding comments and renaming variables to make the code easier to read and maintain. Refer to the Spark Project Overview page and Data Scientist Capstone Project Rubric to make sure you are including all components of the capstone project and meet all expectations. Remember, this includes thorough documentation in a README file in a Github repository, as well as a web app or blog post.


```python

```
