""""
##########################################################################
########################### SYSTEM #######################################
##########################################################################
""""
#############Import all the used libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sqlalchemy import create_engine

import json as json
from pandas.io.json import json_normalize
from jsonschema import validate 

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import re
import os

from datetime import datetime, date, timedelta 
from dateutil import parser

from statsmodels.api import tsa
from fbprophet import Prophet

import warnings
from IPython.display import display
from subspaceutil import generate_data, project_onto_subspace, plot_projection

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer, PolynomialFeatures

from sklearn.compose import ColumnTransformer

from sklearn.base import TransformerMixin, BaseEstimator


from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb

from mlxtend.classifier import StackingCVClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances


from sklearn.tree import export_graphviz
import graphviz

import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *

import pickle
import joblib


from keras.datasets import mnist # api to download the mnist dataset
from keras.models import Sequential # class of neural networks with one layer after the other
from keras.models import Model #when you're using only one layer
from keras.layers.core import Dense, Activation  # type of layers
from keras.optimizers import SGD # Optimisers, here the stochastic gradient descent 
from keras.utils import np_utils # extra tools like the one below here.
from keras.utils.np_utils import to_categorical 
from keras.layers.core import Flatten, Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D #This is for CNN with images
from keras.layers.convolutional import ZeroPadding2D #This is for CNN for images
from keras.layers import Input, Dense ##For CNN (with time series)
from keras.layers import Conv1D #This is for time series with CNN

from keras.layers import SimpleRNN, TimeDistributed #these are helpful for time series
from keras.layers import LSTM
keras.layers import Concatenate #For bidirectional RNN


from keras.preprocessing.image import ImageDataGenerator

import imageio # for elementary image manipulation
import cv2 # for image manipulations


from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string.punctuation as punctuations
from random import shuffle
from sklearn.decomposition import TruncatedSVD

from collections import defaultdict, Counter #1
from numpy import cumsum, sum, searchsorted #2
from numpy.random import rand #3


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt
from lightfm import LightFM
from lightfm.evaluation import recall_at_k, precision_at_k, auc_score, reciprocal_rank


import eli5
from eli5.sklearn import PermutationImportance
from lime.lime_tabular import LimeTabularExplainer
from functools import partial
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import shap
# Need to load JS vis in the notebook
shap.initjs()


import warnings
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', 20)   # pandas: only print out 20 rows rather than the default 60.
np.set_printoptions(precision=2, suppress=True)  # numpy: only print the first 2 decimal places.


%load_ext autoreload
%autoreload 2


os.getcwd() #getwd() in R, pwd in shell


""""
##########################################################################
########################### ZERO PHASE ##################################
##########################################################################
""""

""""
##########################################
##################################### Getting data from relational DBs
##########################################
""""
#Inconvenient to work with many relations

""""
################## SQL
""""

%load_ext sql
%sql postgresql://postgres:Luth1en@2@localhost/postgres

#####Create/fill/delete tables

DROP TABLE tableName;

DROP TABLE IF EXISTS tableName;

CREATE TABLE tableName (
    col1_Name INTEGER PRIMARY KEY UNIQUE NOT NULL, #Or I could also say SERIAL PRIMARY KEY
    col2_Name TYPE, #(eg TEXT or INTEGER)
    col3_Name TYPE, #(eg. FLOAT, r INT, or DATE) 
    col3_Name REAL NOT NULL CHECK (col3_Name > value), #You can inforce rules in teh creation of a table for future inserted values
    col4_Name TEXT REFERENCES anotherTable (colInThatOthertable), #You can also make pointers/links ot other tables
    col5_Name JSONB
);

#####Indexing
CREATE INDEX tableName ON tableName(columnName)



#####Adding values
INSERT INTO tableName VALUES (value1, "string1", value2, "string2");

INSERT INTO tableName (id, name) VALUES (4, 'Dental Care');


#####Transactions
#Transactions allow us to bundle a collection of database operations into a one logical action. This ensures that the state of the database is not updated unless all the logical operations in the transaction are successfully carried out.
#Doing updates without transactions
UPDATE tableName
    SET column_2 = 4
    WHERE column_1 = '321732946'


#Doing updates with transactions
BEGIN TRANSACTION;
UPDATE tableName
    SET column_2 = 5
    WHERE column_1 = '321732946';
INSERT INTO AnothertableName (ColName_1, ColName_2) VALUES(4, 'Dental Care');
COMMIT TRANSACTION;



#####Selection
SELECT * FROM tableName;

SELECT col1_Name, col2_Name FROM tableName;

SELECT * FROM tableName 
    LIMIT 10; #LIMIT 10 is like a head(10)

SELECT colName AS "NewcolName" FROM tableName;




#####Selection with functions
SELECT DISTINCT(columnName) FROM tableName; #gives the unique values

SELECT column1, COALESCE(column2, 'Unknown') AS "column2NewName" FROM tableName; #COALESCE fills the null values with the given string value 'Unknown'

SELECT * FROM tablename
    ORDER BY columnname DESC; #This says, order by this column in descending order. Use ASC for ascending order.



#####Selection with conditions
SELECT columnName FROM tableName WHERE columnName IS NOT NULL;

SELECT column1, column2 FROM tableName
    WHERE (columnName1 IS NULL) AND (columnName2 IS NULL);
    
SELECT column1, column2 FROM tableName
    WHERE (condition_1) AND (condition_2);

SELECT column1, column2 FROM tableName
    WHERE (condition_1) OR (condition_2);

SELECT SUM(ColumnName) FROM tableName WHERE EXTRACT(YEAR FROM deteColumn) = '2016';




#####Selection with conditions for text (regex-like)
SELECT MIN(columnName) AS "smallest ColumnName", MAX(columnName) AS "highest ColumnName" 
    FROM tableName; #You can also use SUM, AVG, COUNT

SELECT colum1, column2 
    FROM tableName
    WHERE TextColumn = 'str value from column1';    

SELECT * 
    FROM tableName
    WHERE TextColumn LIKE '%.co.uk'; #SO in SQL, the % is like the perl *

SELECT * 
    FROM tableName
    WHERE TextColumn IN ('Str1', 'Str2');
    



#####Group by
SELECT colName_1, SUM(colName_2) AS "NewColName" 
    FROM tableName
    GROUP BY colName_1
    ORDER BY NewColName ASC; #Will print the sum of the values in colName_2 grouped by colName_1

SELECT colName_1, SUM(colName2) As "NewColName" 
    FROM tableName
    WHERE colName_1 IN (value1, value2) #If value is a string, but it in quotes ""
    GROUP BY colName_1
    ORDER BY NewColName ASC;

SELECT colName_1, COUNT(*) AS "count" 
    FROM tableName
    GROUP BY colName_1 #To get how many entries of each unique value in colName_1


SELECT colName_1, AVG(amount) AS avg, MIN(amount) AS min, MAX(amount) AS max 
    FROM tableName
    GROUP BY colName_1



#####Conditions on the results of a group by (HAVING)
SELECT colName_1, SUM(colName_2) AS "total" 
    FROM tableName
    GROUP BY colName_1
    HAVING SUM(colName_2) > 100
    ORDER BY total ASC;

SELECT EXTRACT(YEAR FROM dateColumnName) AS "year", SUM(colName_2) AS "total" 
    FROM tableName
    GROUP BY year
    HAVING EXTRACT(YEAR FROM dateColumnName) IS NOT NULL;




#####Subqueries
SELECT col_1, col2 
    FROM tableName AS newTableName
    WHERE newTableName.col_1 
    IN (SELECT col1 FROM AnothertableName)

SELECT MIN(Score) as "min_score", AVG(Score) as "avg_score", MAX(Score) as "max_score" 
    FROM inspections
    WHERE business_id
    IN (SELECT CAST(business_id AS TEXT) FROM businesses WHERE postal_code = '94103')
#CAST(ColName AS newType) when I want to compare a col that is e.g. text to the same col info in another table that has it in e.g. int.
SELECT avg(product::FLOAT) FROM sales; #This ::FLOAT thing is another syntax for casting


#####Unions (like pd.concat(dfs) #where dfs is a list of dfs)
#To aggregate rows from 2 or more queries into a single result (rbind)
SELECT colName 
    FROM tableName
    UNION     #This union by default removed the duplicates
    SELECT colName FROM AnothertableName

SELECT colName 
    FROM tableName
    UNION ALL     #This union by does NOT removed the duplicates
    SELECT colName FROM AnothertableName


SELECT product_id, COUNT(*) AS count 
    FROM (SELECT product_id FROM review UNION ALL SELECT product_id FROM refused_review) AS product_id_union
    GROUP BY product_id 
    ORDER BY count DESC



#####Join: To merge columns (cbind)
#There are different modalities:

##Naural join
SELECT * 
    FROM product as p
    JOIN category as c 
    ON c.id = p.category
    WHERE p.asin = '321732947' OR p.asin = '321732946'

##Cross join:
SELECT * 
    FROM tableName
    CROSS JOIN AnothertableName


##Left-outer join
SELECT col1, col2, col3 
    FROM TableName AS NewTableName
    LEFT OUTER JOIN AnotherTable AS NewAnotherTableName
    ON NewAnotherTableName.reviewer_id = NewTableName.id;



##Inner join
SELECT col1, col2, col3 
    FROM tableName AS NewTableName
    INNER JOIN AnotherTableName AS NewAnotherTableName
    ON NewTableName.col4 = NewAnotherTableName.colName


SELECT r.helpful, r.summary, p.title, p.brand 
    FROM review as r
    INNER JOIN product as p 
    ON r.product_id = p.asin

SELECT r.overall, u.name, r.summary, p.title, c.name as category
    FROM review as r
    INNER JOIN product as p ON r.product_id = p.asin
    INNER JOIN user_account as u ON r.reviewer_id = u.id
    INNER JOIN category as c ON p.category = c.id




""""
################## SQLAlchemy
""""

#####Load/read an already saved DB
import sqlalchemy as db
engine = db.create_engine('sqlite:////Users/marielisandrazepedamendoza/Desktop/CambridgeSpark/Class5/ads_05-databases-v1.0/data/sfscores.sqlite')


#####To make a new DB
from sqlalchemy import create_engine
engine = create_engine("postgresql://postgres:Luth1en@2@localhost/postgres")


#To query a DB
query = "SELECT * FROM tabeName"
result = engine.execute(query)
department_sizes = result.fetchall()


""""
################## SQL with pandas
""""

query = 'SELECT * FROM tabeName'
df_result = pandas.read_sql(query, engine) #It already returns the result in a df


df.to_sql('tableName', engine, if_exists="append", index=False) #to turn a pandas df into an SQL table. Instead of 'append', it could also be 'replace'



""""
##########################################
##################################### Getting data from document oriented DBs
##########################################
""""
#Easy to model one-to-many relations
#Difficult to model many-to-one relations
#Difficult to model many-to-many relations

""""
##################### JSON
""""

#######################JSON with json

#Read the json file
with open('data/fie.json', 'r') as f:
    text_FileData = f.read()  
#print(textFileData) #Will print the whole file
json_FileData = json.loads(text_FileData)
#print('{} looks like an integer'.format(json_FileData['results']['channel']['units']))


#Make a json format doc
dict_object = {'key1': 'value1', 'key2': 'value2'}
json_object = json.dumps(dict_object)


#######################JSON with pandas

df = pd.read_json('data/file.json')

#If there are nested structures, use json_normalize to flatten it
from pandas.io.json import json_normalize
df = json_normalize(json.loads(text_FileData)) #where is text_FileData has the 'jsonTextString'

#Save a df to a json file
with open('output.json', 'w') as text_file:
    text_file.write(df.to_json(orient='records'))

#Validate a json file
from jsonschema import validate 
json_loads = json.loads('{'key1': 'value1','key2': 'value2'}')
schema = {
    'type': 'object',
    'properties': {
        'key1': {'type': 'string'}
        'key2': {'type': 'number'} #or {"type" : "array", "items": {"type": "string"}}, or smtg else
    }
}
validate(json_loads, schema)


#######################JSON with postgres

#So, this is to deal with json entries in an SQL db

%load_ext sql
from sqlalchemy import create_engine
engine = create_engine("postgresql://postgres:Luth1en@2@localhost/postgres")

DROP TABLE IF EXISTS students;
CREATE TABLE students (
    id serial primary key,
    details JSONB 
);

#as an example, details has 
INSERT INTO students (details) VALUES ('{"name": "Rupert", "city": "London"}');

#Queries on sqls on json cols
DELETE FROM students 
    WHERE details @> '{"city":"London"}'

SELECT details ->> 'name' FROM students 
    WHERE details @> '{"city":"London"}'

SELECT details ->> 'name' FROM students
    WHERE details ->> 'city' = 'London'

SELECT details ->> 'title' FROM books WHERE detail ->> 'title' LIKE '%effect%'

SELECT COUNT(1) FROM books 
    WHERE details @> '{"type": {"key":"work"}}'


SELECT column_id, jsonb_array_elements(detail -> 'payload' -> 'commits') AS commit
    FROM ghevents


SELECT COUNT(DISTINCT column_id)
    FROM (SELECT jsonb_array_elements(details -> 'payload' -> 'commits') as commit, column_id) as commits
    WHERE commits.commit -> 'author' ->> 'email' LIKE '%co.uk%'



#Update an attribute from the json
UPDATE students 
SET details = jsonb_set(details, '{"city"}', '"Cambridge"')
    WHERE details @> '{"name":"Rupert"}';






""""
##########################################
##################################### Getting data from graphs DBs
##########################################
""""
#The relations are first-class citizens
#Nodes contain key:value pairs of simple values
#Easy one-to-many, many-to-one and many-to-many relations


""""
##################### CYPHER
""""

%load_ext cypher
# Open the Neo4j Desktop application and click create a new graph put the passowrd bellow, and click Start
# change the password (after the ':') as required. 
# Here username is 'neo4j'
# The pwd is 'Zrwk3vPDex'
# The database name is 'data'
%config CypherMagic.uri='http://neo4j:Zrwk3vPDex@localhost:7474/db/data'

#################Read a csv file into a Neo4j graph
##########From a file without headers
#cat artists.csv
#1,ABBA,1992
#2,Roxette,1986
#3,Europe,1979
#4,The Cardigans,1992
LOAD CSV FROM 'https://neo4j.com/docs/cypher-manual/3.5/csv/artists.csv' AS line
CREATE (:Artist { name: line[1], year: toInteger(line[2])})
#Here a new node with the Artist label is created for each row in the CSV file. In addition, two columns from the CSV file are set as properties on the nodes.

##########From a file with headers
#Id,Name,Year
#1,ABBA,1992
#2,Roxette,1986
#3,Europe,1979
#4,The Cardigans,1992
LOAD CSV WITH HEADERS FROM 'https://neo4j.com/docs/cypher-manual/3.5/csv/artists-with-headers.csv' AS line
CREATE (:Artist { name: line.Name, year: toInteger(line.Year)})

LOAD CSV FROM 'https://neo4j.com/docs/cypher-manual/3.5/csv/artists-fieldterminator.csv' AS line FIELDTERMINATOR ';'
CREATE (:Artist { name: line[1], year: toInteger(line[2])})  #If the delimeter was a ; instead of a ,

#A more complicated example
USING PERIODIC COMMIT  #This line is if we're importing a large file
LOAD CSV WITH HEADERS FROM "https://s3-eu-west-1.amazonaws.com/csparkdata/clinton-emails.csv" AS line
MERGE (fr:Person {alias: COALESCE(line.MetadataFrom, line.ExtractedFrom, '')}) #MetadataFrom/To, ExtractedFrom/To/Cc are headers of the csv file
MERGE (to:Person {alias: COALESCE(line.MetadataTo, line.ExtractedTo, '')})
MERGE (cc:Person {alias: COALESCE(line.ExtractedCc, '')})
MERGE (fr)-[:SENT_EMAIL_TO]->(to)
MERGE (fr)-[:CC_EMAIL_TO]->(cc)


#################Create a graph
#Ensure you have a clean db
match (n)-[r]-() delete n, r

#Create nodes of a db
CREATE (nodeValue1:NodeTypeName {NodeInfoKey1:"NodeValue1", NodeInfoKey2:value2, NodeInfoKey3:"value3"})
#examples
CREATE (rupert:Person {firstname: "Ruper", age: 24, city:"London"})
CREATE (travelling:Hobby {name: "Travelling"})

#Create edges of a db
CREATE (node1)-[:EdgeNameType]->(node2)
#example
CREATE (rupert)-[:LIKES]->(travelling) 


#################Queries
MATCH (n) RETURN n #get all the nodes

MATCH (a)-[r]-(b) RETURN a, type(r), b  #type(r) gives you the name/type of the edge. Get all the connections in the graph

MATCH (rupert:Person {firstname: "Rupert"}) --> (person:Person) RETURN person #Get all the Person nodeTypes' info to which rupert is connected, regardless of the EdgeNameType, not caring aout that EdgeNameType info

MATCH (rupert:Person {firstname: "Rupert"}) -[r]-> (person:Person) RETURN rupert, type(r), person

MATCH (rupert:Person {firstname:"Rupert"}) -[:LIKES]-> (person:Person) RETURN person

MATCH (rupert:Person {firstname:"Rupert"}) --> (person)-[:LIKES]->(hobby) RETURN person, hobby

MATCH (n)-[:SENT_EMAIL_TO]->(n) RETURN COUNT(n)

MATCH (n)-[:SENT_EMAIL_TO]->(p)-[:SENT_EMAIL_TO]->(n) RETURN n, p  #find cycles


#################Queries with conditions

#More than one matching element, an AND condition (conditions are separated by a ,)
MATCH 
    (n)-[:SENT_EMAIL_TO]->(p), 
    (n)-[:CC_EMAIL_TO]->(n) WHERE n.alias <> "" #<> is the comparison operator of inequality. Basically the same as != 
    RETURN n as sender, p as receiver, n as cc

MATCH 
    (p), 
    (n:Person {alias: "Clinton, Hillary"}) WHERE NOT (p)-[:SENT_EMAIL_TO]->(n) 
    RETURN p



""""
##########################################
##################################### Getting data from pictures
##########################################
""""

import imageio # for elementary image manipulation
import cv2 # for image manipulations


# specifies the default figure size for this notebook
plt.rcParams['figure.figsize'] = (10, 10)
# specifies the default color map
plt.rcParams['image.cmap'] = 'gray'

img = cv2.imread('data/cat.jpg')
#or
picture = imageio.imread('data/grace_hopper.jpg', as_gray=True)
#Show the image
plt.figure()
plt.imshow(picture)
plt.show()





""""
##########################################
##################################### Getting made up data
##########################################
""""


np.linspace(-10, 10, 500) #Get 500 points between [-10, 10]

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

rash = np.random.rand(10000) > 0.90 #10% of people have a rash
high_temp = np.random.rand(10000) > 0.99 #1% of people have high_temperature

np.zeros(10000)




""""
##########################################################################
########################### FIRST PHASE ##################################
##########################################################################
""""

""""
##########################################
##################################### EDA
##########################################
""""

"""
#################Handle non-time-series data
"""

#Import data from an excel or csv or table txt file
df = pd.read_excel('data/tfl-daily-cycle-hires.xls', sheet_name='Data')
df = pd.read_excel("data/default.xls", skiprows=1)

df = pd.read_csv('data/retail_data.csv', index_col="CustomerID", na_values=[' ?', '?', '? ', 'NA', 'missing'])

df = pd.read_table('data/retail_data.txt', header=None)

#Import data from an object (e.g. a pickled dataframe)
df = pandas.read_pickle('data/peninsula_publicpay_gzip.pickle', compression='gzip')

#Import data from a JSON file with pandas
df = pd.read_json('data/file.json')




df = raw_data.copy()


#Get a sense of the data
df.head(5)
df.shape
df.describe(include='all')
df.dtypes
df[df.isnull().any(axis=1)] #To print rows with missing values
null_columns=df.columns[df.isnull().any()] #Get the columns with null values
df.apply(pd.isnull).sum() #investigate which features have NaNs and how many
df.drop(columns='time_between_orders') #Delete a particulat column
#this can also be writen as: 
del df['time_between_orders']
df.dropna() #Delete all rows that have a NaN value
df.columns
df.index
df['Country'] #Get the series of a particular column
df[["Distance", "DepDelay"]] #Get more than one particular column
df.Country #Get the series of a particular column
df.iloc[0] #Get the row of a particular index
df.loc['index string name'] #Get the row of a particular index by its ID
df.count() #By column, count the number of non NaN values
df.quantile(0.25)
df.min()/df.max()
df.idxmin()/df.idxmax()
df.applymap(f) #Apply function to each element in the df
df.reset_index()
df.replace("?", np.nan)
df.columnName.replace({' <=50K.': ' <=50K', ' >50K.':' >50K'}) 
df.nunique() #number of unique values per each column
df.index.nunique() #Get the number of unique index elements
df.agg(['mean', 'count', 'min', 'max', 'sum']) #Like the describe, but you can specify which stats you want only
df.isnull().sum() #Get number of nulls per col

df.Country.value_counts(normalize=True) #equivalent of a "sort | uniq -c" of the column "Country", but more powerful because it normalized teh counts already. 

df['total_spent'].groupby(df['Country'])
df.groupby(['tag'])['amount'].agg(['mean', 'count', 'sum'])


pd.cut(df["n_orders"], [0, 1, 2, 5, 10, np.inf]) #assign values of a column to ranges


#queries
df.query('n_orders <= 25')
df.query('total_refunded < 0 and total_refunded > -1000')

df.Country[df.Country != 'United Kingdom'] #is the same as the next:
df.query('Country != "United Kingdom"')

#And for arrays, use the np.where function
y_pred[np.where(y_pred >= threshold)] = 1

df.sort_values("column_to_sort_by", ascending=True, inplace=True)

map(function, vector) #it's like apply but for vectors/lists

# Delete an item from dictionary 
del[Dictionary1['C']] 

#Delete specified index positions from a list:
vector = [0,1,2,3,4]
indexesToDelete = [1,4]
np.delete(vector, indexesToDelete)
#[1,2,3] would be the result


#Make little functions with lambda:
checker = lambda x: (np.mean(x), np.var(x))
checker(x)


"""
#################Handle time-series data
"""

#To read a time series dataset setting the date col as index in pandas TimeStamp format
df = pd.read_csv('data/bikes.csv', parse_dates=['day'], index_col='day')
#Or set day as index after reading it normally
df.set_index(pd.to_datetime(df.day), inplace=True)
df.drop("day", axis=1, inplace=True)
#pd might have a bug saying that it does not plot with periods. You just gotta specify the x axis like this:
plt.plot(df.index, df)


#There's 3 main diff packages to deal with time series:

#######Python datetime package
current_date = date.today()
current_date_time = datetime.now()
new_date = datetime(year=2018, month=10, day=13)
#This package goes in hand with the dateutil.parser module to obtain datetime objects from parsing strings
new_date = parser.parse("13th October 2018")
new_date.year
new_date.month
new_date.day
new_date_and_time = parser.parse("2018-10-13T15:53:20")
new_date_and_time.hour
new_date_and_time.minute
new_date_and_time.second
#Arithmetics with time
NOW = datetime.now()
DELTA = timedelta(days=14)
print(NOW + DELTA)


#######Numpy datetime64 type
current_date_time = np.datetime64(datetime.now()) #to convert from python datetime object to np datetime64
new_date = np.datetime64("2018-11-03")
new_date_and_time = np.datetime64("2018-10-03 12:00")
np_array_of_dates = np.array(['2018-11-02', '2018-10-02', '2015-11-03'], dtype='datetime64') #create an array of numpy datetime64 objects
#The numpy API doesn't provide ways to access the different parts of a datetime (year, hour, minute etc). You will need pandas to do this.
pd.to_datetime(new_date).year
##Arithmetics with time
np.datetime64("2018-11-03") - np.datetime64("2018-11-01")
np.datetime64('2018-11-03') + np.timedelta64(14, 'D')
np.datetime64('2018-10-03 12:00') + np.timedelta64(6, 'h')
np.datetime64('2018-11-03') + np.arange(10) #This gives you an array of days 2018-11-03 to 2018-11-12. Stated like this, the addition is made to the lowest unite of the datetime64 object. In this case days.
np_array_of_dates + np.timedelta64(7, 'D') #This adds 7 days to each entry in np_array_of_dates


#######Pandas Timestamp type
pd.Timestamp.now()
date = pd.to_datetime("14th of October, 2018")
date = pd.Timestamp(2018, 10, 14)
date = pd.Timestamp(year=2018, month=10, day=14, hour=12, minute=0, second=30)
date.day
date.dayofyear
date.week
date.month
date.year
date.hour
date.minute
date.second
date.strftime('%c')  #To go from a timeStamp to a spelled out day with day of week, number, month and year
#To transform to the python datetime
python_datetime = date.to_pydatetime()
#To work with time series in a pandas series as indexes
datetimes = pd.DatetimeIndex(['2014-07-04', '2014-08-04', '2015-07-04', '2015-08-04'])
series = pd.Series([10, 4, 14, 30], index=datetimes)
series['2015'] #Gives you the power to access them as if a query. This gives you all the entries with indexes from this year. would work the same for a df.
series.plot() #And the plot is smart adjusting the time spans in the x axis
#Create range of dates. The default step is by day
dates = pd.date_range('2015-07-03', '2015-07-10')
weeks = pd.date_range('2018 Oct 1', periods = 10, freq = 'W')
#Arithmetics
weeks - weeks[0]
#Time-zones
time_utc = pd.Timestamp.now(tz="UTC") #It's better to have times in UTC and then at the end tz_convert if needed
london = pd.Timestamp.now(tz="Europe/London")
brussels = london.tz_convert("Europe/Brussels")
#To work with time series in a pandas dataframe or series as indexes
df = pd.read_csv('data/bikes.csv', parse_dates=['date'], index_col='date')
time_interval_start = pd.Timestamp("1st January 2012")
time_interval_end = pd.Timestamp("31st January 2012")
df_specificTimeInterval = df[time_interval_start:time_interval_end]
#Resampling: We can aggregate time series by resampling the points on a coarser time level. 
df = df.resample('M').mean() #Take all the entries by month and get the mean of each feature. You can resample by day, week or month (not by year, to do by year it's resampling by 12 months)
df = df.resample('12M').mean()
#Parse custom date formats directly when loading the df file
df = pd.read_csv('data/NZAlcoholConsumption.csv', 
                                  parse_dates=['DATE'], #The column with the custome date format
                                  date_parser=parse_quarter, #A previously defined function that processes 'DATE' and returns a pd.Timestamp
                                  index_col='DATE')
df.sort_index(inplace=True)



##########Things to look at/do in time series

#Rolling averages by time windows to help identify trends
df_mean = df.ColumnOfInterest.rolling(window=4).mean()
interact(rolling_avg_plot, window_size=(0, 10)) #where rolling_avg_plot is a func that plots the rolling mean of a given window. interact is a widget function
#Differencing: looking at the time series formed of differences between values separated by a given lag. To identify the length of the seasonality
df.ColumnOfInterest.diff(1) #lag of one day. You should remove the seasonality and work with stationary time series data. This could also be considered part of preprocessing I guess.
#Autocorrelation: measures the correlation (similarity) between the time series and a lagged version of itself. Because if there's seasonality in the data, it has to be removed. So keep the lag with the highest autocorr value.
df.ColumnOfInterest.autocorr(lag=lag)








"""
#################Handle image data
"""

import cv2 # for image manipulations

img = cv2.imread('data/cat.jpg')

img = cv2.resize(img, (224, 224))
plt.imshow(imageio.imread('data/cat.jpg')); plt.axis('off');
plt.figure()


#It could be that the neural network assumes that the input it will receive is an array of a given size (say 224 by 224 RGB) - hence using the resize command above -. Also, it might assume that the images have been *centered* in each of their channels. 
# This transformation performs the 0-centering
def transform_image(image):
    image_t = np.copy(image).astype(np.float32) # Avoids modifying the original
    image_t[:, :, 0] -= 103.939                 # Substracts mean Red
    image_t[:, :, 1] -= 116.779                 # Substracts mean Green
    image_t[:, :, 2] -= 123.68                  # Substracts mean Blue
    image_t = np.expand_dims(image_t, axis=0)   # The convolutional neural network takes batches of images as input. The first dimension (the one we just added with this np.expand_dims) is a "dummy" dimension, that is because the network expects an array of images as input.
    return image_t

img_t = transform_image(img)
print(img_t.shape)
#(1, 224, 224, 3) #The first dimension is a "dummy" dimension, that is because the network expects an array (i.e. a batch) of images as input. The three subsequent dimensions are the image dimensions with the three colour channels at the end. 





"""
#################Handle text data
"""

##The "read.delim" equivalent
file=open(filename)
fileContent=file.readlines()
file.close()
#And then to go line by line:
for fileLine in fileContent:
	fileLine=fileLine.rstrip() #The perl chomp equivalent

## The while read line equivalent:
target = open(filename)
for line in target:
	do stuff
target.close()

## Another way of reading a file, this one with while
file = open(filename)
line= file.readline()
while line:
	do smth
	line= file.readline()

##Put in a directory the content of all the read files in a directory (reads the file all at once, instead of line by line):
import os
def init_dict(a_dir):
    a_dict = {}
    file_list = os.listdir(a_dir)
    for a_file in file_list:
        f = open(a_dir + a_file, 'r')
        a_dict[a_file] = f.read()
        f.close()
    return a_dict


###Writing output files
out= open("outputFile.txt", 'w')
for rxn in result:
	out.write("%s\t%s\t%s\n" % (rxn , str(result[rxn]['minimum']) ,  str(result[rxn]['maximum'])    )

out.close()

#Some common str-related commnds          
re.compile('\w+').findall(line)
re.sub('[^0-9a-zA-Z\s]+', '', review)
x1.lower()
x1.upper()


"""
#################Handle big data
"""

##############PySpark RDDs

#Make a context 
import pyspark
from pyspark import SparkContext
sc = SparkContext.getOrCreate()
print(sc)
print("Ready to go!")

#Create RDD with parallelize
python_data = ["We", "love", "Coding"]
data = sc.parallelize(words)
#Or if it's text:
data = sc.textFile("data/war-and-peace.txt")
#And if the txt file was jsn with one entry per line:
data_json = sc.textFile("data/HNStories.json")
data = data_json.map(lambda x: json.loads(x))
data.persist() #To cache the resulting RDD


###Basic RDD manipulation

#count numberof entries
data.count()
#Print first X lines (the head() equivalent)
for line in data.take(15):
    print(line)
#Filter
data.filter(lambda line: "war" in inhouseFuct_toGetWords(line))
#map
data.map(lambda line: line.upper())
#flatmap
data.flatMap(lambda line: get_words(line))
#distinct
words.distinct()
#Set-lke transformations
warLines.union(peaceLines)
warLines.subtract(peaceLines)
warLines.intersection(peaceLines)
data.filter(lambda line: "war" in get_words(line) and "peace" in get_words(line))

#Key/Value pairs
word_pairs = words.map(lambda word: (word, 1))
word_pairs.reduceByKey(lambda c1, c2: c1 + c2)
word_pairs.groupByKey().map(lambda pair: (pair[0], len(pair[1])))


##############PySpark DataFrames

#Make a session
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate() 

#Read data sources for later sql/dataframe stuff
reviews_paths =['data/reviews_fashion.json', 
                'data/reviews_electronics.json',
                'data/reviews_sports.json']
data_sets = spark.read.json(reviews_paths)
#Or
POSIX = "data/reviews_*.json"
data_sets = spark.read.json(POSIX)

#Build data to a schema
from pyspark.sql.types import *
REVIEWS_SCHEMA_DEF = StructType([
        StructField('reviewerID', StringType(), True), #the True means that it is nullable (i.e. it can contain nulls)
        StructField('asin', StringType(), True),
        StructField('reviewerName', StringType(), True),
        StructField('helpful', ArrayType(
                IntegerType(), True), 
            True),
        StructField('reviewTime', StringType(), True),
        StructField('overall', DoubleType(), True),
        StructField('summary', StringType(), True),
        StructField('unixReviewTime', LongType(), True)
    ])
data_sets = spark.read.json(POSIX, schema=REVIEWS_SCHEMA_DEF)
#Print hte schema
data_sets.printSchema()

######DataFrame operations:
#DataFrame API have functionality similar to that of Core RDD API. 
#For example:
#map : foreach, Select
#filter : filter
#groupByKey, reduceByKey : groupBy

#Selecting columns
select_df = reviews.select(reviews.asin, reviews.overall,
                           reviews.helpful[0]/reviews.helpful[1],
                           reviews.reviewerID)
                    .withColumnRenamed('(helpful[0] / helpful[1])','helpful')
select_df.show()
#Filtering
select_df.filter(select_df.overall >= 3)
select_df.where(select_df.helpful >= 0.5)
#Grouping
select_df.groupby(select_df.overall).count()

#Set-like operations
df1.join(df2, df2.asin == df1.asin)
         .dropna(subset="title")

#Missing values
#Drop nans
df.dropna(subset=["price"])
#Fill nans
from pyspark.sql.functions import mean, count, max
average = df.select(mean(df.price))
new_df = df.fillna(average, subset=["price"])

#To apply to all columns
df.apply(lambda x: x.fillna(x.mean()),axis=0)


################ Use Spark SQL

# Register the DataFrames to be used in sql
df.createOrReplaceTempView("df")

#Query
sql_query = """SELECT reviews.asin, overall, reviewText, price
            FROM reviews JOIN products ON reviews.asin=products.asin
            WHERE price > 50.00
"""
result = spark.sql(sql_query)
result.show()

#User Defined Functions
def some_function(arg):
    smtg stmg
    return smtgArrayTypeWithStringType
#1. register the dataframe as a table
product_df.createOrReplaceTempView("products")
#2. register the UDF
spark.udf.register("some_function", 
                   some_function, 
                   returnType=ArrayType(StringType()))
#3.use the UDF in SQL 
sql_query_UDF = """SELECT asin, title, price, some_function(categories) as UDFresult
                        FROM products
                    """
result = spark.sql(sql_query_UDF)
result.show(truncate=False)


####Spark UI and Data skewness
partitioned_data_rdd = spark.sparkContext.parallelize(data, 2)  #Here I specify that I only want two partitions
partition_rdd.getNumPartitions()
partition_view = rdd_obj.mapPartitions(lambda l: [l]).map(list).collect()

#To access the Spark UI go to http://localhost:4040
partition_rdd.reduceByKey(lambda a, b: a + b, numPartitions=5)


############## Parquet 

####Using Pyarrow

import pyarrow as pa
import pandas
import pyarrow.parquet as pq

loans = pandas.read_csv("data/loan.csv")
table = pa.Table.from_pandas(loans)

pq.write_table(table, "data/loan.parquet", compression='none')
table_loan = pq.read_table("data/loan.parquet")
table_loan.to_pandas().head()

####Using Spark

import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

booksDF = spark.read.json("data/openlibrary.json")

booksDF.filter(booksDF['title'].contains("Cambridge")).count()

booksDF.write.parquet("data/books-spark.parquet",mode='overwrite')

rddBooks.map(json.loads)\
.filter(lambda book: "title" in book)\
.filter(lambda book: "Cambridge"in book["title"])\
.count()




"""
#################Plots
"""

#Spagetti plot
df.plot(kind='line')
plt.legend(loc='upper left')
plt.savefig('plot1.pdf')

# Histogram
sns.countplot(df.Country)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Class', data=ccfd, ax=plt.gca())
plt.xticks([0, 1], ["Non-Fraud", "Fraud"], fontsize=12)
plt.yticks([1e5, 2e5], ["0.1", "0.2"], fontsize=12)
ax.set_ylabel("Number of occurences in millions", fontsize=14)
## to label the column with value:
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 2000, '{}'.format(height), ha="center") 

plt.show()


#Histogram from a pandas series
ratings['rating'].value_counts(sort=True, ascending=True).plot(kind="bar")

plot = dtf.plot()
fig = plot.get_figure()
plt.tight_layout() #So tha tthe x axis labels are not cut out of the saved pdf
fig.savefig("output.png")


PdSeries.plot(kind="bar")

df['quality'].hist()

#Barplot of the values of a column in relation to the labels of another column
sns.barplot(x='Country', y='sum of total spent', data=df)

# Distribution of number of orders
sns.distplot(df['n_orders'], rug=True)
plt.xlabel("Time (normalised)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

#Violin plots
sns.violinplot(x="Country", y="total_spent", inner="quart", data=df)

#Boxplots
sns.boxplot(x="Country", y="total_spent", data=df)
sns.boxplot(data) #Of a single column/vector

#Heatmaps
sns.heatmap(df, linewidths=0.5, cmap="RdBu", vmin=-1, vmax=1, annot=True)

#Scatter plots
plt.scatter(x, y)

sns.pairplot(df)  #This can help seeing the correlations present in your data
sns.pairplot(df, vars=cols, hue='cluster')
pd.plotting.scatter_matrix(df, figsize=(8,8)) #It's the same as the sns.pairplot, but the sns one is better



plt.figure()
# you could use plt.scatter in a similar fashion
plt.plot(crosses[:,0], crosses[:,1], marker='x', linestyle='none', label='cross')
plt.plot(circles[:,0], circles[:,1], marker='o', linestyle='none', label='circle')
plt.legend()
plt.ylim((0, 2))
plt.xlim((0, 6))
plt.show()


plt.figure()
plt.ylabel('Probabilities')
plt.xlabel('Categories')
plt.vlines(data1, [0], out, label='out1', color='C0')
plt.vlines(data2, [0], out, label='out2', color='C1')
plt.legend()
plt.show()



plt.figure()
plt.plot(X_train['Time'], label='Train')
plt.plot(X_test['Time'], label='Test')
plt.axhline(X_train['Time'].max(), color='k', linestyle='--', alpha=.5, label='Max time in train set')
plt.ylabel('Time variable value')
plt.xlabel('Datapoint index')
plt.legend()
plt.show()


#Plot a roc curve
plt.figure(figsize=(8, 6))
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = {0:.2f})'.format(auc(fpr, tpr)))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') # random-guess baseline
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Receiver Operating Characteristic', fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.show()


#Heatmaps on a pandas dataframe
df.style.background_gradient()


#Barplot of a feature grouped by another feature (so, more than one bar per feature)
fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(x='Survived', data=train, hue='Sex')
ax.set_ylim(0,500)
plt.title("Impact of Sex on Survived")
plt.show()

              

melted_series = pd.melt(series, id_vars="Date", var_name="source", value_name="publications")
sns.catplot(x='Date', y='publications', hue='source', data=melted_series, kind='bar')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #To put the legend box outside of the plot fig

              

              
fig, ax = plt.subplots()
fig.set_size_inches(13.7, 8.27)
Low_income_rates.plot(ax=ax)
ax.set_xticks(range(0,len(range(2001,2018))+1))
ax.set_xticklabels(range(2001,2018), rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
fg = ax.get_figure()
fg.savefig("PapersPerYear_PerCountry_Low_income_rates_line.pdf")


Stacked barplot using a pivot df
pivot_df.loc[:,['Jan','Feb', 'Mar']].plot.bar(stacked=True, color=colors, figsize=(10,7))
 
              
              
              
              
              
              
              
"""
#################Project onto subspace
"""

#####With project_onto_subspace

#1.- Project into subspace to find hidden structures in the data to explore it a lot deeper. You can project x and y onto a line rotated in diff angles to see if you find any hidden data structures. 
X -= X.mean(0) #gotta standardize X to have mean 0
A = lambda theta: np.array([np.cos(theta*np.pi/180), np.sin(theta*np.pi/180)]).reshape(2,1)  #The function that defines the subspace matrix
projected_X, X_new_coods = project_onto_subspace(A(45), X)
#Plot the projection to find any interesting patterns
plot_projection(X, projected_X)
plt.hist(X_new_coods, bins=40);
#2.- Another reason you might wanna project into a subspace is to reduce dimensions finding the projection that maximizes the variance (although for this just doing a linear regression would be the best)
#Calculate the variance of the X_new_coords over a wide range of angles, and try to find the subspace that maximises the variance.
np.var(X_new_coods)



#########t-SNE 

#t-Distributed Stochastic Neighbor Embedding (t-SNE): This is an unsupervised model algorithm, but it's actually more to be used for EDA and 'communication purposes' (aka, making pretty plots). 
#It's like PCA BUT it does not transform linearly, and though it's a dimensionality reduction technique, it's not used for reducing the complexity of your dataset to feed it to your model to make predictions (like the PCA can be used for)

tsne = TSNE(n_components=2, perplexity=40, verbose=2)
tsne = tsne.fit_transform(df)



""""
####################
#####Preprocess data
####################
""""

#After we've removed any NaNs columns during the EDA, there's other processing we can do:

#If your data has categorical features, we must represent them numerically in order to run any analysis. The standard approach is via a one-hot encoding
neighbourhood = pd.get_dummies(df['neighbourhood'], prefix='neighbourhood') #The output is a df with the columns with the one hot encoding. You merge them with:
new_df = pd.concat([df, neighbourhood], axis=1)
#Deal with missing values. They can be informative or not. You might wanna remove the column or impute the missing values. 
rental_df.isnull().sum() #See how many missing values per column

df.dropna() #Remove rows with missing data

#Imputation with a summary statistic such as mean or median or most freq val or any other decided value
df['columnsWithSomeNaNs'].fillna(0) #Here we're filling NaNs with 0

#Remove outlayers (a number of Machine Learning techniques are sensitive to outliers)
def remove_outliers(data, k=3):
    mu       = data.mean() # get the mean
    sigma    = data.std()  # get the standard deviation
    filtered = data[(mu - k*sigma < data) & (data < mu + k*sigma)]
    return filtered
df = df.apply(remove_outliers)
df = df.dropna()

#Scaling. Algorithms often assume that all features are centered around zero and have variance in the same order. So, center and scale your data, so that all the dimensions fall onto a comparable interval.
scaler = StandardScaler() 
df_scaled = pd.DataFrame(scaler.fit_transform(df),
                         columns = df.columns,
                         index = df.index)
#This is supposed to be the same as using hte StandardScaler fit_transform
X_scaled -= X.mean(0); X_scaled /= X_scaled.std(0)


#Identify correlated features (by getting a correlation matrix)
corrmat = df.corr()
#And to get the correlation coefficient between two features:
np.corrcoef(f1, f2)[0, 1]

#Dimensional reduction by variance thresholding. If you have many many features, you might wanna consider removing features with low variance (we are examining the variance for a given feature across samples)
sel = VarianceThreshold(threshold=0.5)
sel.fit(df)
columns_to_keep = sel.get_support() #return an array of True/False for which columns pass the threshold
new_df = df.iloc[:, columns_to_keep]


#Check for colinearity. When a feature could be expressed with high accuracy by a linear combination of other features, the resulting model can be unstable and generalise badly. A simple way to test for colinearity is to compute the condition number. If the number is much higher than 1000, either apply PCA to the X to use the PCA transformed data as input. Or use regularization at the step of runing LogisticRegression.
np.linalg.cond(X)


#Save your processed data into a new txt file
df.to_csv('data/processed_data.csv', index=False)





""""
##########################################
########Advanced data preprocessing
##########################################
""""



"""
###############Transformers
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer, PolynomialFeatures
from sklearn.base import TransformerMixin, BaseEstimator


########FeatureUnion

numerical_transformer = FeatureUnion([
    ('log', FunctionTransformer(np.log)),
    ('polynomial', PolynomialFeatures(2, include_bias=False, 
                                 interaction_only=True))])

#df = X_train[["AGE", "LIMIT_BAL"]]
numerical_transformer.fit_transform(df)


########Custom transformer

#You can make one by making a class that uses `TransformerMixin` and `BaseEstimator`, because those are required to define a new transformer in the right format for sklearn.

class CustomScaler(TransformerMixin, BaseEstimator):
    def fit(self, X, y= None):
        self.median = np.median(X, axis= 0)
        self.interquartile_range = interquartile_range(X)
        return self
    def transform(self, X, y= None):
        Xt = (X - self.median) / self.interquartile_range
        return Xt


# Instanciate the transformer
inhouse_transfomer = CustomScaler()
# Fit
inhouse_transfomer.fit(X_train)
print(inhouse_transfomer.median)
# Transform
inhouse_transfomer.transform(X_train)



########ColumnTransformer

preprocessor = ColumnTransformer([("DoNothing", "passthrough", doNothing_cols), #Use passthrough for the ones I don't want to do anything
                                  ("payments", payment_transformer, payment_cols), 
                                  ("numerical", numerical_transformer, num_cols),
                                  ("unpaid", unpaid_transfomer, unpaid_cols),
                                  ("categorical",OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_cols)])

preprocessor.fit(X_train)
preprocessor.transform(X_train)

#When we call transform, the preprocessor returns a numpy array, which is great for Machine Learning algorithms to process ... but not so great for us humans to interpret. So we will wrap it up back to DataFrame with nice column names. To do so we will need to give nice names to the dummy feature generated by the one hot encoder. The code below extracts the list of categories, creates nice names for the dummy feature and create a new list all_features with good names for our columns

ohe_categories = preprocessor.named_transformers_["categorical"].categories_
new_ohe_features = [f"{col}__{val}" 
                    for col, vals in zip(cat_features, ohe_categories) for val in vals]
all_features = doNothing_cols + payment_cols + num_cols + unpaid_cols + new_ohe_features

X_train = pd.DataFrame(preprocessor.transform(X_train), columns=all_features)
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=all_features)


#Note on one-hot-encoding, you can use maps if you are doing ordered encoding of features. e.g.
modeTransport_map = {'car': 0, 'walk': 1, 'bike': 2, 'pt': 3}
df.modeTransport = df.modeTransport.map(mode_map)



"""
###############Pipeline
"""

#You can define sequential unchangable sets of transformations into a pipeline (which can be put into a transformer

#In a pipeline, the transformations are sequential

payment_transformer = Pipeline([("scaler", MinMaxScaler()), ("pca", PCA(n_components=0.8))])
#df = X_train[["PAY_6", "BILL_AMT6", "PAY_AMT6"]]
payment_transformer.fit_transform(df)


#If you want it to fit it to the train set and use the fited one on the test set (using the pipeline's memory):
pipeline = Pipeline([('scaling', StandardScaler())])
preprocessor = pipeline.fit(X_train)
X_train_s = preprocessor.transform(X_train)
X_test_s = preprocessor.transform(X_test)



#You can also tune the hyperparameters of the pipeline (which can include the classification model) with a gridSearch. For example:

logreg = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression())])
#Imagine preprocessor is our transformer that preprocesses the data and uses PCA
logreg.named_steps
#Tune the hyperparameters for the preprocessor and the model elements/steps of the pipeline
params = {'preprocessor__payments__pca__n_components': [0.6, 0.8],
          'model__penalty': ['l1', 'l2'],
          'model__C': [0.001, 0.01]}
gcv = GridSearchCV(logreg, param_grid=params)
gcv.fit(X_train, y_train)
print(gcv.best_params_)
#Set the model tuned parameters into the pipeline
logreg.set_params(**gcv.best_params_)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print(classification_report(y_test, y_pred_logreg))





""""
##########################################################################
########################### SECOND PHASE #################################
##########################################################################
""""


""""
##############################################################
###################### UNSUPERVISED MODELS 
##############################################################
""""


""""
##################################
############################# PCA
##################################
""""

#PCA can also be considered part of EDA, but it should be done after processing the data, or can be used as a tool to guide the processing of the data as well. It also can be considered an unsupervised model. 
pca = PCA()
pca.fit(df_scaled)

#Get the W (that has the "loading" values) of the fitted model
W = pca.components_ #the PCs are the rows, the features are the columns
#Or if you want it as a pd dataframe
W_df = pd.DataFrame(pca.components_, 
                        columns = df_scaled.columns,
                        index   = pca.columns)

#To retrieve the new coordinates (for PCA is a projection of the data onto a new space)
A = pca.fit_transform(df_scaled)
#Or if you want to keep it into a pd dataframe:
A_df = pd.DataFrame(
                        pca.fit_transform(df_scaled), 
                        columns = ['PC'+str(i+1) for i in range(df_scaled.shape[1])], 
                        index   = df_scaled.index
                        )

#Get how much variance each PC explains
pca.explained_variance_ratio_
#Get the cumulative explained variance
cum_expl_ratio = np.cumsum(pca.explained_variance_ratio_)

#Plot your first 2 components
plt.scatter(A[:,0], A[:,1])
sns.lmplot(x='PC1', y='PC2', data=A_df, fit_reg=False)


#so, for dimensionality reduction, PCA for dense data or TruncatedSVD for sparse data




""""
##################################
############### K-means clustering
##################################
""""

kmeans = KMeans(n_clusters = 2)
kmeans.fit(df)
kmeans.cluster_centers_ #Look at the centroids of the clusters 
cluster_assignment = kmeans.predict(df)
df['cluster'] = cluster_assignment
sns.pairplot(df, vars=cols, hue='cluster') #Plot it


""""
##################################
########## Hierarchical clustering
##################################
""""

##Two packages for doing it:

####With scipy (the cool thing is that it has this very cool dendrogram function)
Z = linkage(df, method='ward', metric='euclidean')
# Draw the dendrogram
dendrogram(Z)
dendrogram(Z, color_threshold=cut_off, truncate_mode='lastp', p=20)
plt.hlines(cut_off, 0, len(customers), linestyle='--')


####With sklearn
agglomerative = AgglomerativeClustering(n_clusters=3, linkage='ward', affinity='euclidean')
cluster_assignment = agglomerative.fit_predict(df)
#Here you can only visualize it with the sns.pairplot, not with a dendrogram like when using the scipy implementation.





""""
##################################
################ DBSCAN clustering
##################################
""""

#You need to specify a min number of samples to initiate a cluster, and a radious of search (eps parameter)
#1.- Find hte best eps value:
all_distances = pairwise_distances(df, metric='euclidean')
# get the distance of each point to its closest neighbor
neig_distances = [np.min(row[np.nonzero(row)]) for row in all_distances]
#2.- Visualize it to decide on the best value
plt.hist(neig_distances, bins=50)
#3.- Now, fit_predict the model
db = DBSCAN(eps=1.0, min_samples=8)
cluster_assignment = db.fit_predict(df)





""""
##################################
############################## NLP
##################################
""""


"""
###############Clustering
"""

#Unsupervised learning for when you want to explore the inherent relations between the data points in your dataset


#0.- First, split the text data
from random import shuffle
shuffle(data) 
#Or, if it's in a dataframe, with the labels together with teh text:
df = df.sample(frac=1).reset_index(drop=True)
###Preprocess input text

#1.- Tokenize, lemmatize, make lower case, and remove stop words and punctuation marks.
      
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
import string
punctuations = string.punctuation

def spacy_tokenizer(text):
    tokens = nlp(text)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return (' '.join(tokens))

#1.2.- Apply the function to train and test datasets
data = [spacy_tokenizer(text) for text in data]


#2.- TfidfVectorizer and  TruncatedSVD
              
from sklearn.decomposition import TruncatedSVD #We'll use this one for dimensionality reduction, instead of PCA.

#TfidfVectorizer is equivalent to CountVectorizer followed by TfidfTransformer
def transform(orig_data, orig_dim, red_dim):
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.5,
                                 max_features=orig_dim,
                                 stop_words='english',
                                 use_idf=True) #1
    trans_data = vectorizer.fit_transform(orig_data)

    print("\nOriginal data contains: " + str(trans_data.shape[0]) +
          " with " + str(trans_data.shape[1]) + " features")

    svd = TruncatedSVD(red_dim) #2
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    return lsa.fit_transform(trans_data), vectorizer, svd

trans_data, vectorizer, svd = transform(data, 10000, 300)

#YOu can explore your dataset with, e.g. looking at the vocabulary
vectorizer.vocabulary_
vectorizer.vocabulary_.keys()

#3.- Apply clustering algorithm

km = KMeans(n_clusters=2)

km.fit(trans_data)

#4.- Evaluation: If we have the classifications of the points we can evaluate the clusters. 
#- Homogeneity – a measure that evaluates if each of the clusters contains only data points from a single class (note the similarity to precision for classification algorithms).
#- Completeness – a measure that tells if all the data points from a particular class are elements of the same cluster (note the similarity to recall for classification algorithms).
#- You may note that homogeneity and completeness are complementary to each other. V-measure is the harmonic mean between the two that allows to take both into account

gs_clusters = np.unique(labels)

def evaluate(km, labels, svd):
    print("Clustering report:\n") #1
    print("* Homogeneity: " + str(metrics.homogeneity_score(labels, km.labels_)))
    print("* Completeness: " + str(metrics.completeness_score(labels, km.labels_)))
    print("* V-measure: " + str(metrics.v_measure_score(labels, km.labels_)))

    print("\nMost discriminative words per cluster:")
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names() #This bit extracts the 50 most discriminative words per cluster.
    for i in range(gs_clusters):
        print("Cluster " + str(i) + ": ")
        cl_terms = ""
        for ind in order_centroids[i, :50]: #2
            cl_terms += terms[ind] + " "
        print(cl_terms + "\n")
              

evaluate(km, labels, svd)


              

"""
###############Language modelling and language generation
"""

from collections import defaultdict, Counter
from numpy import cumsum, sum, searchsorted
from numpy.random import rand

#1.- Define the functionality of the language model. 

class LanguageModel(object):
    def __init__(self, order=1): #1
        '''Initializes a language model of the given order.'''
        self._transitions = defaultdict(int) #2
        self._order = order
    
    def train(self, sequence): #1
        '''Trains the model using sequence.'''
        self._symbols = list(set(sequence))
        for i in range(len(sequence)-self._order):
            self._transitions[sequence[i:i+self._order], sequence[i+self._order]] += 1 #2

    def predict(self, symbol): #3
        '''Takes as input a string and predicts the next character.'''
        if len(symbol) != self._order:
            raise ValueError('Expected string of %d chars, got %d' % (self._order, len(symbol))) #4
        probs = [self._transitions[(symbol, s)] for s in self._symbols]
        return self._symbols[self._weighted_pick(probs)] #5

    def generate(self, start, n): #6
        '''Generates n characters from start.'''
        result = start
        for i in range(n):
            new = self.predict(start) #7
            result += new
            start = start[1:] + new #8
        return result

    @staticmethod
    def _weighted_pick(weights): #9
        '''Weighted random selection returns n_picks random indexes.
        The chance to pick the index i is given by weights[i].'''
        return searchsorted(cumsum(weights), rand()*sum(weights))



#2.- Apply the model

model = LanguageModel(order=8)
model.train(training_corpus)
print (model.generate('Lisandra', 100)) #You ask the model to generate a sequence of 100 characters with the starting characters 'Lisandra'


#3.- Evaluation

#The most widely used measurement of the language model performance is called _perplexity_, and it is based on measuring how probable in the language the piece of text generated by a language model is. Let's implement this measure and evaluate the text generated by the language model above.

nlp = spacy.load('en_core_web_sm')
tokens = nlp(training_corpus)
print(len(tokens))

#It would be reasonable to evaluate whether it generates a sequence of legitimate English words. For that, you can use word unigrams and estimate whether the language model based on character prediction generates highly probable English words.
# In practice, it is a good idea to reserve some small amount of probability for the words unseen in the training data, e.g.  𝜆=0.001 . 

def unigram(tokens):    
    model = defaultdict(lambda: 0.001)
    for token in tokens:
        token = str(token).strip()
        try:
            model[token] += 1
        except KeyError:
            model[token] = 1
            continue
    total = 0
    for word in model:
        total+= model.get(word)
    for word in model:
        model[word] = model.get(word)/float(total)
    return model


def perplexity(testset, model):
    testset = nlp(testset)
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[str(word).strip()])
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

#the better the language model, the lower the perplexity value
              
model = unigram(tokens)
lm = LanguageModel(order=8)
lm.train(training_corpus)
testset2 = lm.generate('the best', 280)
print(testset2)
print(perplexity(testset2, model))



              
              

"""
###############Word2vec
"""

###################################
#######Simplest implementation
#######for simple one string input
###################################

import spacy
# nlp = spacy.load('en')
# nlp = spacy.load('en_core_web_md')
nlp = spacy.load('en_core_web_sm') #This is an already made English model

text = u'cat dog apple orange pasta pizza coffee tea'
words = nlp(text)

print("\t" + text.replace(" ", "\t"))
# Print out word similarities in a table
for word1 in words:
    output = str(word1) + "\t"
    for word2 in words:
        output += str(round(word1.similarity(word2), 4)) + "\t"
    print(output)


statement = u'Cat and dog are similar. Cat and frog aren\'t.'
text = nlp(statement.lower())
cat = text[0]
dog = text[2]
frog = text[8]
print ("Is statement \"" + statement + "\" correct?")
print(cat.similarity(dog))
print(cat.similarity(frog))
print(cat.similarity(dog) > cat.similarity(frog))

#You can also do this similarity thing between entire corpuse, not just between specific words within a corpus: 
#review1.similarity(review2)


###################################
#######Analogy task
###################################

cosine = lambda v1, v2: np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

words = nlp(text) #Text is a string with Countries and Capitals.

question = u"Russia is to Moscow as China is to WHAT?"
text = nlp(question)
source1 = text[0]
source2 = text[3]
target1 = text[5]

max_sim = 0.0
target2 = "N/A"

#Apply the operations on vectors
target2_vector = source2.vector-source1.vector+target1.vector

#Find the word with the most similar vector to the result
for word in words:
    if not (str(word)==str(target1) or str(word)==str(source1) or str(word)==str(source2)):
        current_sim = cosine(target2_vector, word.vector)
        if current_sim >= max_sim:
            max_sim = current_sim 
            target2 = word

print(question)
print(target2)


###################################
#######Simplest implementation
#######for already made model
###################################


# add your code here to define a function cosine that takes two vectors and returns the similarity
cosine = lambda v1, v2: np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
#So with this lambda you can calculate the distance between two words from your word2vec model    
word1_vec = w2v_model['word1']
word2_vec = w2v_model['word2']

cosine(word1_vec, word2_vec)



###################################
#######Make model
###################################

w2v_model = Word2Vec(min_count=20,
                     window= EMBEDDING_SIZE,   #e.g. 5
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)


w2v_model.build_vocab(sentences, progress_per=10000)  #where sentences are the cleaned and tokenized text

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

#As we do not plan to train the model any further, we are calling init_sims(), which will make the model much more memory-efficient:
w2v_model.init_sims(replace=True)

##Exploring the model
w2v_model.wv.most_similar(positive=["resistance"])




       
              

""""
##########################################################
####################### SUPERVISED MODELS 
##########################################################
""""

#Label encode to 0s and 1s the feature that tells you to which class your data point (your sample) belongs to
le = LabelEncoder()
le.fit(df['ClassColumn'].unique())
y = le.transform(df['ClassColumn'])
#Extract your feature matrix. Meaning, the X to work with. So, delete the ClassColumn
X = df.drop(columns='ClassColumn')

#You could also use map instead, and do smtg like:
y = df["ClassColumn"].map({"no": 0, "yes": 1})
X = df.drop("ClassColumn", axis=1)



"""
############### Non-hierarchical data
"""

##################Split data

#Split your data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=321)
#Or, if most of your labels are form one class, you got to stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)

##################Cross-validation

#Once you've decided on the model(s) to try, you might want to optimize your hyper-parameters using GridSearchCV. For an example on how to do it, see the Logistic Regression section

clf = LogisticRegression()
C_range = 2. ** np.arange(-5, 5, step=1) 
parameters = [{'C': C_range, 'penalty': ['l1', 'l2']}]
grid = GridSearchCV(LogisticRegression(), parameters, cv=5, 
                    scoring='accuracy', return_train_score=True) 
#In the gridsearch, you could also use scoring="balanced_accuracy", in case there is a class imbalance in your dataset.

grid.fit(X_train, y_train)
clf_cv = grid.best_estimator_

print(grid.best_params_)
print(grid.best_score_)


"""
############### Hierarchical data
"""

##################Split data

#When you have multiple entries from the same entity. E.g.
#"So we can clearly see that people from the same household seem to be making multiple trips, and they will be highly correlated, in terms of explanitary variables (e.g. green, the percentage of green space in the vicinity around the household), and the target variable (e.g. mode choice). For instance, four trips are made a 24 year old male in household 3460, all walking, and all with identical explanitary variables, except distance and density."

#So what is the relevance of the hierarchical nature of the data?
#When we sample data randomly to form a test set, each individual row has an equal probability of being selected, which means there is a high chance of rows (trips) made by the same household appearing in both the test and train dataset. This is bad, as we can then overfit the model to noise in that households data - i.e. unique features of the household/individual, and not general relations which indicate someone is likely to take one mode over another.
#If you don't do a proper split, that will introduce correlations between test and train data which are specific to the household, and not general.

#We deal with hierarchical data when sampling test data by doing grouped sampling.

hh = df.household_id
gss = model_selection.GroupShuffleSplit(n_splits=1, train_size=0.7, 
                                        test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=hh))

X_train_g = X.iloc[train_idx]
X_test_g = X.iloc[test_idx]
y_train_g = y[train_idx]
y_test_g = y[test_idx]

hh_train_g = hh.iloc[train_idx]
hh_test_g = hh.iloc[test_idx]

##################Cross-validation

#Use GroupKFold with 3 splits and GridSearchCV to search max_depth valus of 2, 6 and 12.
gcv = model_selection.GroupKFold(n_splits=3).split(X_train_g, y_train_g, groups=hh_train_g)
gs = model_selection.GridSearchCV(clf, param_grid=params, 
                                  scoring=['accuracy','neg_log_loss'],
                                  n_jobs=1, cv=gcv, refit='neg_log_loss', verbose=3)
gs.fit(X_train_g, y_train_g)
clf_cv = gs.best_estimator_



""""
##################################
############### Linear regression
##################################
""""

###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%From scratch
#Initialize your w array to the size of the number of features in your X plus 1 (for the intercept).
w = np.zeros(2)
#Then, that w0 that is for the intercept is removed from the model by addinf to X a column of 1s.
X = np.hstack((np.ones((200, 1)), x))
#The predicted value y_pred is Xw 
y_pred = X @ w
#And a loss function that calculates the loss of our predctions y_pred (meaning, the residuals) using the squared L2 norm is to be minimized 
r = y - y_pred
loss = np.linalg.norm(r)**2
#Then you get the gradient of the loss function (-X.T @ r )
grad = -X.T @ r
#Then you optimize your loss function doing gradient descent. Ths will give you the modeled values of w.
simpleGD(w, loss_Xy, gradient_Xy, 1e-6, 1000)
#Finally plot your line
plt.scatter(x, y)
plot_abline(w[1], w[0], '--', color='red')
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Predict the values of y with lr. sklearn has everything already to do a lr model.
lr = LinearRegression() 
lr.fit(X_train, y_train)   #This x is supposed to be the training x
y_pred =  lr.predict(X_test) #This x is meant to be the test x

#Calculate the model loss in the dataset
L_y = np.linalg.norm(y_test - y_pred)**2

""""
##################################
############## Logistic regression
##################################
""""

###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%From scratch
#A logistic regression model returns a value between 0 and 1 which we can interpret as the probability of being in class 1.

##So, what we want is to get the values of w. First, initialize w by as long as the number of columns in X plus 1 (for the intercept, the bias term).
w = np.zeros((X.shape[1]+1))

#Append a column of 1's to your X (for the intercept, the bias term)
X = np.hstack([np.ones((X.shape[0], 1)), X])

#Get the sigmoid function
def sig(x):
    output = 1/(1 + np.exp(-x))
    return output

#Get the logistic regression function
def logistic_regression(X, w):
    a = X @ w    #remember X@w was the formula for linear regression
    y_pred = sig(a)
    return y_pred

#Calculate the loss using the cross-entropy 
def cross_entropy(y, y_pred):
    loss = -(1/y.shape[0]) * np.sum( np.log( y_pred**y * (1 - y_pred)**(1-y)   )  )
    return loss

def loss_function(w, X, y):
    # calculate y_pred using predicted_values
    y_pred = logistic_regression(X, w)
    # calculate the cross-entropy loss from y and y_pred
    loss = cross_entropy(y, y_pred)
    return loss

#Get the gradient function
def gradient(w, X, y):
    y_pred = logistic_regression(X, w)
    N = y.shape[0]
    grad = 1/N * X.T @ (y_pred - y)
    return grad

#Use gradient descent to get the values of w (to optimize)
def simpleGD(w0, f, g, gamma, nr_steps):
    history = np.zeros(nr_steps+1)
    history[0] = f(w0)
    w = w0
    for ii in range(nr_steps):
        # this formulation amounts to writing w = w - stepsize*g(w)
        w -= gamma*g(w)
        history[ii+1] = f(w)
    return w, history

#Get the values of w
w, hitory = simpleGD(w, loss_Xy, gradient_Xy, 1e-2, 10000)

#Optimization diagnostics. If you see that the loss increases over your iterates, reset w and optimize again with a smaller step size.
plt.plot(history)
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#If the condition number in your colinearity check is much higher than 1000, you can use regularization in the step where the LogisticRegression() is instanciated. To this, use the params penalty (l1 or l2) and C. The C parameter is the inverse of the regularisation strength. In other words, lower C means more regularisation.
#Find the best values for C and penalty using cross validation
C_range    = 2. ** np.arange(-5, 5, step=1) #It is standard to use regularisation strengths on a logarithmic scale
# build a dictionary of parameters
parameters = [{'C': C_range, 'penalty': ['l1', 'l2']}]
grid = GridSearchCV(LogisticRegression(), parameters, cv=5, return_train_score=True) #use 5 cross validation sets
grid.fit(X_train, y_train)
#Get the best hyperparameter values
grid.cv_results_
grid.cv_results_['mean_test_score'] #mean_test_score of the 5-fold cross-validation
bestC = grid.best_params_['C']
bestP = grid.best_params_['penalty']

#Apply the logistic regression like this:
logreg = LogisticRegression(penalty=bestP, C=bestC)
logref.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
#Or using the grid
y_pred = grid.predict(X_test)

#You can get the probability of each prediction for belonging to class 1
y_pred_probs = logreg.predict_proba(X_test)[:,1]





""""
##############################################
####################### SVC
##############################################
""""

#Support Vector Classifier is a kind of linear model

from sklearn.svm import SVC

penalty = 0.1
svc = SVC(C=penalty)  #You could also run it without specifying a penalty and using the default one. You allow more misclassification by lowering the value of C.
svc.fit(X_train, y_train)

print("Accuracy score on test set: {0:.4f} (C={1:.2f})".format(accuracy_score(y_test, svc.predict(X_test)), penalty))

#Note that if the problem is not a binary classification problem (3 classes). The SVC is extended by default to a multi-class classifier using the one-vs-rest (OvR) scheme.

#As the SVM determines a set of weights for the linear model, it is possible to investigate these, to determine whether the model is looking at a large set of features, or just is able to determine the class based on a small set of features.

plt.figure(figsize=(8, 6))
coefs = [svc.coef_[0, i] for i in range(X_train.shape[1])]
sns.distplot(coefs, kde=False)


################ SVC using kernels

#By applying the kernel method to SVM, a wider classes of datasets can be modelled (including those that are not linearly separable but may be non-linearly separable). By using different kernels, it is possible to obtain non-linear decision boundaries, which might be a better fit depending on the dataset.
#Note that in case of already high-dimensional using a kernel may lead to overfitting.

#svc = SVC(kernel='linear', C=penalty)
svc_rbf = SVC(kernel='rbf', C=0.5, gamma=0.5)
svc_rbf.fit(X_train, y_train)

#com+ add your code here to display the accuracy and plot the regions
print("Accuracy score on test set: {0:.4f}".format(accuracy_score(y_test, svc_rbf.predict(X_test))))

#When faced with both a linear model and a non-linear model that perform well, it is often better to pick the least complex model (Occam's Razor). In this case, the linear model.






""""
##################################
############################# GNB
##################################
""""

#Gaussian Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("#Mislabeled points (NB): {1} out of {0}".format(len(y_test), sum(y_pred != y_test)))




""""
##################################
############################# KNN
##################################
""""

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train) 

#Get the predictions
y_pred = knn.predict(X_test) #Result is a boolean vector

#You can get the probability of each prediction for belonging to class 1
y_pred_probs = knn.predict_proba(X_test)[:,1]

#Get the accuracy
knn.score(X_train,y_train)
knn.score(X_test,y_test)
#If model's accuracy is similar across both training and test data sets, it is a good indication that the model has not overfitted on the training set: if it were, we would expect a substantial drop in performance on the test set as the model failed to generalise to unseen data points.


""""
##################################
############################# DTC
##################################
""""

#Fit the model
dtc = DecisionTreeClassifier(max_depth=3)
dtc.fit(X_train, y_train)

#Visualize the DTC
dot_data = export_graphviz(dtc, out_file=None, 
    feature_names=X_train.columns,  
    class_names=['not delayed', 'delayed'],  
    filled=True, rounded=True,  
    special_characters=True, rotate=True)
graph = graphviz.Source(dot_data)
graph
#You can also use pydotplus for visualizing it
import pydotplus  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())




#Get the predictions
y_pred = dtc.predict(X_test)

#You can get the probability of each prediction for belonging to class 1
y_pred_probs = dtc.predict_proba(X_test)[:,1]

#Get the accuracy of the model
dtc.score(X_train,y_train)
dtc.score(X_test,y_test)





""""
##################################
############################# DTR
##################################
""""

#The Decision tree above is a decision tree clasifier, here is the decision tree regressor. The idea is similar to classification trees, but the leaves contain a scalar value instead of a class label.

from sklearn.tree import DecisionTreeRegressor

#Initialize the model, train it and get the predictions
dtr = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)

#Get the accuracy of the model
dtc.score(X_train,y_train)
dtc.score(X_test,y_test)


#While decision trees offer a lot of flexibility in terms of the model, this flexibility comes at a significant cost. In order to reduce overfitting, either hyper parameter optimisation is required, yet a more common approach is to resort to ensembles. Emsembles are explained below.


""""
##############################################
####################### Ensemble models 
##############################################
""""


"""
###############Random Forest
"""

#As kind of bagging

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=250, random_state=0)
y_pred = clf.fit(X_train, y_train).predict(X_test)
print("#Mislabeled points (RF): {1} out of {0}".format(len(y_test), sum(y_pred != y_test)))


#Note that a Random Forest also provides an indication of the importance of the different features. This is because the different decision trees in the Random Forest will all select different features to construct the model, based on the randomly selected data and features they observe. Based on the decisions in the different models, it is possible to estimate which of the features are most important. Plot the feature importance for the previous model below. Use the feature_importances_ attribute for this.

pd.DataFrame(clf.feature_importances_, 
             index=X_train.columns, 
             columns=["importance"]).sort_values("importance", ascending=False).plot(kind="bar")



#Since DTCs are very sensitive to the starting parameters, use a gridSearch to fine tune the hyperparameters
param_grid = {"max_depth": [3, 5],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10]}
# run grid search
grid_search = GridSearchCV(RandomForestClassifier(n_estimators=75), 
                           param_grid=param_grid, n_jobs=-1)
grid_search.fit(X_train, y_train)
#Check the results
grid_search.cv_results_
grid_search..best_params_


"""
###############XGBoost 
"""

#A kind of boosting
#In Gradient Boosted trees, different decision trees are built sequentially. While the first tree is built to predict the original dataset, the subsequent trees focus on correcting the errors of the last ensemble.

#xgboost has a built-in way to deal directly with missing values, so we can keep them in.

#Gradient Boost are typically senstive to hyperparameter optimisation, so we will immediately attempt to optimise the model using CV. `n_estimators` is the number of trees to build. Getting this parameter right is really important as too many boosting trees will lead to overfitting. A more extensive study of the hyperparameters is given in http://xgboost.readthedocs.io/en/latest/parameter.html


import xgboost as xgb

cv_params = {'max_depth': [5, 7], 'min_child_weight': [1, 3], 'n_estimators': [50, 100]}
ind_params = {'learning_rate': 0.1, 'random_state':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
              'objective': 'binary:logistic'}

optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                             cv_params, scoring='accuracy', 
                             cv = 5, n_jobs = -1) #n_jobs = -1 means use all your  computer cores for the calculations
optimized_GBM.fit(X_train, y_train)

#Check best params
print(optimized_GBM.best_params_)

#Predict
y_pred = optimized_GBM.predict(X_test)

#Similar to Random Forest, XGboost allows you to see the importance of your features. Use `xgb.plot_importance`
plt.figure(figsize=(20,15))
xgb.plot_importance(optimized_GBM.best_estimator_, ax=plt.gca())
 




"""
###############LightGBM 
"""

#Another gradient boosting algorithm

import lightgbm as lgb

cv_params = {'max_depth': [5, 7], 'min_child_weight': [1, 3], 'n_estimators': [50, 100]}
ind_params = {'learning_rate': .1, 'random_state':0, 'subsample': 0.8, 'colsample_bytree': 0.8}

optimized_LGB = GridSearchCV(lgb.LGBMClassifier(**ind_params), cv_params, scoring='accuracy', cv = 5, n_jobs = -1) 
optimized_LGB.fit(X_train, y_train)

y_pred = optimized_LGB.predict(X_test)

#Examine the features importance
plt.figure(figsize=(20,20))
lgb.plot_importance(optimized_LGB.best_estimator_, ax=plt.gca())







"""
###############Stacking
"""

from mlxtend.classifier import StackingCVClassifier

#The basic usage is of the form:
#stack = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
#                           meta_classifier = clfmeta,
#                           use_features_in_secondary=True)



l0_rf = RandomForestClassifier()
l0_knn = KNeighborsClassifier(n_neighbors=2)
l0_gnb = GaussianNB()
l1_lr = LogisticRegression(penalty='l2')
stack = StackingCVClassifier(classifiers=[l0_rf, l0_knn, l0_gnb],
                           meta_classifier=l1_lr,
                           use_features_in_secondary=True)

stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(np.array(train.label))
y_test = le.transform(np.array(test.label))

print(classification_report(y_test, y_pred, target_names=le.classes_))  #<---- because the sklearn prints the classed targets, not the names


##Run with doing a gridserach 

params = {'kneighborsclassifier__n_neighbors': [1, 5],
         'randomforestclassifier__n_estimators': [100, 150],
         'meta_classifier__C': [0.1, 0.5]}

gcv = GridSearchCV(estimator=stack, 
                   param_grid=params, 
                   cv=10, n_jobs=-1)

gcv.fit(X_train, y_train)

#y_pred = gcv.best_estimator_.predict(X_test)
y_pred = gcv.predict(X_test)  #This one is the same as the one above, it's just that the one above is more explicit/verbose.

print(classification_report(y_test, y_pred))






""""
##############################################
################### Multiclass classification
##############################################
""""

"""
Some algorithms are inherently multiclass:
● KNN
● DTC and RF

But the ones that are inherently not (Support Vector Classifier, 0/1 Logistic Regression), can be made so with some tricks explained next
"""

""""
################################################
#######Autoregression (for time series modeling)
################################################
""""

#In an autoregression model, values are modelled as a linear combination of the  𝑝  past values. 
#There are 2 ways which you can use to do this:

#####TSA simple AR
ar = tsa.AR(time_series)
optlag = ar.select_order(10, ic="aic") #To identify the best lag value
ar_result = ar.fit(maxlag=optlag)
prediction = ar_result.predict(start=optlag)

#####TSA ARMA 
#Like the AR, but it adds the past errors as additionnal features (the q parameter)
arma = tsa.ARMA(time_series, order=(optlag, 3))  #(3,3) are the p, q parameters
arma_result = arma.fit()
prediction = arma_result.predict(start=optlag)

#To predict into the future:
ar_result = arma_result.predict(start=3, end = time_series.shape[0]+1)  #This will predict 1 point into the future
#Don't use this models to extrapolate too far ahead, because it works the guess to make the second guess


####You can also use the input data to train a model dividing it into train and test like for the other kinds of models of non-time series data
train = time_series[:training_size]
test = time_series[training_size:]
ar = tsa.AR(train)
ar_result = ar.fit(maxlag=4)
prediction = ar_result.predict(end=len(time_series))[-len(test):]   #This is the equivalent of saying XX time points into the future and you keep only those forecasted points
# compute the MAE:
mae = mean_absolute_error(time_series.values[training_size:], prediction)



""""
##########################################
########Prophet (for time series modeling)
##########################################
""""


# The prophet package expects input as a dataframe with the first column indicating time and the second indicating the time series we wish to forecast
# It also expects these columns to have the names 'ds' and 'y'. ds is the pd.DatetimeIndex
#If you have outlayers, mark them as holidays and pass them as another df with teh columns called 'holiday' and 'ds'

forecast_model = Prophet( growth='linear',  weekly_seasonality=3, 
                         yearly_seasonality=3, holidays=all_holidays_strikes )
forecast_model.fit(dat)
#To forecast with the model (in this case 1 year into the future)
df_dates = forecast_model.make_future_dataframe(periods=365, include_history=True)
model_predictions = forecast_model.predict(df_dates)
forecast_model.plot_components( model_predictions, uncertainty=False )




""""
##############################################
############################## Neural networks 
##############################################
""""

"""
############### Vanilla NN (example for pictures)
"""
#NN = Neural network

from keras.datasets import mnist # api to download the mnist dataset
from keras.models import Sequential # class of neural networks with one layer after the other
from keras.layers.core import Dense, Activation # type of layers
from keras.optimizers import SGD # Optimisers, here the stochastic gradient descent 
from keras.utils import np_utils # extra tools

#1.- Display the pictures dataset
plt.imshow(images_train[1234,], cmap="gray")
print(labels_train[1234])

#2.- We now need to reshape the dataset as Keras' architecture expects to get flattened vectors not square matrices as input. Also it expects float32
N_OF_PIXELS = PIXELwidth*PIXELheight
images_train = images_train.reshape(N_TRAINING_SAMPLES, N_OF_PIXELS) 
images_test = images_test.reshape(N_TEST_SAMPLES, N_OF_PIXELS)

images_train = images_train.astype('float32') 
images_test = images_test.astype('float32')

images_train /= 255 # normalising on (0,1) 
images_test /= 255 # normalising on (0,1)

#3.- Tell keras how many categories/classes (N_CLASSES) you have
labels_train = np_utils.to_categorical(labels_train, N_CLASSES)
labels_test = np_utils.to_categorical(labels_test, N_CLASSES)


#4.- Declare the Multi Layer Perceptron (MLP) architecture.
#   1.- declare an instance of Sequential call it model
model = Sequential()
#   2.- add a Dense layer with 500 (number chosen just...because XD) neurons, the input is a vector of N_OF_PIXELS components
model.add(Dense(500,input_shape=(N_OF_PIXELS,)))
#   3.- add an Activation layer with relu units to use on the nodes of that first layer
model.add(Activation('relu'))
#   4.- add another Dense layer with 300 (number chosen just because) neurons
model.add(Dense(300))
#   5.- add another Activation layer with relu units
model.add(Activation('relu'))
#   6.- add a final Dense layer with N_CLASSES neurons
model.add(Dense(N_CLASSES))
#   7.- add a final Activation layer with softmax units
model.add(Activation('softmax'))


#5.- Declare the optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

#6.- Fit the model
model.fit(images_train, labels_train,
          batch_size=100, #number of instances per noisy gradient
          epochs=10, #a measure of computational effort in terms of how many "full gradients" the computational effort amounts to (knowing that each full gradient does a complete pass over the data)
          verbose=2, #whether or not we want to show output during the learning
          validation_data = (images_test,labels_test))




"""
############### CNN (for image modeling)
"""

#Convolutional Neural Networks

import cv2 # for image manipulations
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

#the misc element from scipy will allow us to do some elementary image manipulation
#Note: it's good practice to have the weights in your filter sum to 0 and don't forget to re-arrange the dimensions with  filter.transpose(1, 2, 0)

#######0.- Preprocess pictures
#1.- Turn our images into floating point numbers
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#2.- Normalise the values by 255
X_train /= 255
X_test  /= 255
#3.- Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test  = np_utils.to_categorical(y_test, 10)

##4.- More formally, the image pre-processing tasks are:
# 0-mean across all images ("feature-centering")
# variance 1 across all images ("normalization")
# introduce a random horizontal and vertical shift to create more ("perturbed") training samples (makes the NN more robust as well)
# randomly shift images horizontally
# All this can be implemented in the next transformer: (see more about it in https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/):
datagen = ImageDataGenerator(
        featurewise_center=True,                 # set input mean to 0 over the dataset
        samplewise_center=False,                 # set each sample mean to 0 3Here we set False becaus eI already did that above  by dividing / 255
        featurewise_std_normalization=True,      # divide inputs by std of the dataset
        samplewise_std_normalization=False,      # divide each input by its std
        rotation_range=0,                        # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,                   # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,                  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,                    # randomly flip images
        vertical_flip=False)                     # randomly flip images
datagen.fit(X_train)




############Now, make the model/create the model architecture
#1.- Create the model, it's a Sequential model (stack of layers one after the other)
model = Sequential()

#2.- On the very first layer, you must specify the input shape
#ZeroPadding2D adds a frame of 0 (column left and right, row top and bottom)
#the tuple (1, 1) indicates it's one pixel and symmetric.
model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3))) 
#3.- Your first convolutional layer will have 64 3x3 filters, 
# and will use a relu activation function
model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
#Actually, I could write these two lines as only one like this:
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu')) #32 instead of 64 if it's a 32x32 image

#4.- Stack layers
# Once again you must add padding
model.add(ZeroPadding2D((1, 1)))
#And add the layer
model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))

#5.- Add pooling layers
# Add a pooling layer with window size 2x2
# The stride indicates the distance between each pooled window
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#6.- Add another set of convolutions that go like:
# Padding - Conv - Padding - Conv - Pooling
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1')) #(3, 3) is the patch/kernel size.
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#But the third set would have 256 instead of 128, the fourth would have 512, and the fifth would alos have  512
#As you can see, the depth of the layers get progressively larger, up to 512 for the latest layers. 
#This means that, as we go along, each layer detects a greater number of features. 
#On the other hand, each max-pooling layer halves the height and width of the layer outputs. 


#7.- Add as many more sets as you want. Repeat step 6.

#8.- Add fully connected layers 
#Fully connected layers can learn the more abstract features of the image. But first you must first change the layout of the input so it looks like a 1-D tensor (vector).
# Flatten the output
model.add(Flatten())
#The Flatten function removes the spatial dimensions of the layer output, it is now a simple 1-D vector of numbers. This means we can no longer apply 2D convolution layers as before, but we can apply fully connected layers (Dense layers).
# In ths case, we add a fully connected layer with 4096 neurons
model.add(Dense(4096, activation='relu'))

#9.- Prevent overfitting adding a Dropout layer
model.add(Dropout(0.5)) #The number 0.5 indicates the amount of change, 0.0 means no change, and 1.0 means completely different.
#Dropout is a method used at train time to prevent overfitting. As a layer, it randomly modifies its input so that the neural network learns to be robust to these changes.


#10.- Add one more fully connected layer (and another dropout layer too):
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

#11.- Finally a softmax layer to predict the categories with N_CLASSES (the number of categories to predict) neurons
model.add(Dense(N_CLASSES, activation='softmax'))

#12.- THAT was for setting the architecture of the network. Now, load the weights (if you're already given them:
model.load_weights("pathToTheWeightsFile.h5")

#13.- Compile the network
sgd = SGD() #Initialize teh optimizer #It could also be adam instead of SGD
model.compile(optimizer=sgd, loss='categorical_crossentropy') #Compile with the optimizer
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#14.- Train the model
batch_size = 32 #or 64 if it's a 64 image
nb_epoch = 200
model.fit_generator(
    datagen.flow(X_train, Y_train, batch_size=batch_size), #See how here we use the previouly defined datagen to process the input image
    steps_per_epoch=X_train.shape[0]/batch_size,
    epochs=nb_epoch,
    validation_data=(X_test, Y_test))

#15.-  Predict. Push the (preprocessed - image 0-centered - see code above in the image processing section) image through the network.
out = model.predict(img_t)

#16.- The output is for a batch of images but if you only gave one (like in this case) so extract the first element
out = out[0]   

#17.- # now plot the output, xlabel=Categories, ylabel=Probabilities
plt.figure()
plt.ylabel('Probabilities')
plt.xlabel('Categories')
picture = np.arange(out.shape[0])
plt.vlines(picture, [0], out, label='out', color='C0')
plt.legend()
plt.show()

#18.- Let's look at its top 5 guesses for each of the two images.
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
top_5 = out.argsort()[-5:][::-1]
top_5_values = np.sort(out)[-5:][::-1]
print('Image 1')
for label, prob in zip(labels[top_5], top_5_values):
    print('label: {} with probability: {:0.3f}'.format(label, prob))

#19.- Explore the output if you want. E.g. get the weights of the first convolutional layer 
# (so that's the second layer, just after the input layer)
first_layer_weights = vgg_model.layers[1].get_weights()
# first_layer_weights[0] stores the connection weights of the layer
# first_layer_weights[1] stores the biases of the layer




"""
############### CNN (for time series modeling)
"""

#You can analyze time series with CNNs by representing the time series as an "image".

#When making time series model (maybe not specifically with CNN) consider these:
#   How many modes are there? (in the distplot distribution you made to visualize the data)
#   We do not know the time span of the period we are looking at
#   There could be different seasonality during each of the distinct time periods
#   We could be evaluating on a different distribution (i.e. the test set does not bear much statistical resemblance with the training set), this will lead to poor generalisation.

##### 0.- Prepare time series data for CNN application:
#The following function reshapes a matrix into a number of batches (smaller matrices with fewer rows and the same number of columns). 
def reshape_to_batches(matrix, batch_size):
    # pad the matrix with zeros if the number of rows is not divisible by the batch_size
    # np.ceil is the upper-rounding operator so np.ceil(4.3) == 5.0
    batch_num = np.ceil(matrix.shape[0] / batch_size)
    modulo = batch_num * batch_size - matrix.shape[0]
    if modulo != 0: # not divisible by batch_size
        # add some 0-rows to the matrix
        padding = np.zeros((int(modulo), matrix.shape[1]))
        matrix = np.vstack((matrix, padding))
    return np.array(np.split(matrix, batch_num))

#Reshape the data:
NUMBER_OF_BATCHES = 100
X_train_s_batch = reshape_to_batches(X_train_s, NUMBER_OF_BATCHES)

#If you're going to use the categorical_crossentropy loss function (the standard loss for binary classification), we need to transform our class labels into a binary matrix of (1s and 0s) of shape (samples, classes).
y_binary = to_categorical(y_train)
#And also reshape this into batches
y_train_batch = reshape_to_batches(y_binary, NUMBER_OF_BATCHES)


####### Make the model

from keras.layers import Input, Dense, Conv1D #We use 1D Conv since we are only going to stride one way (along the time axis). 
from keras.models import Model

#1.- Define the input layer
inputs = Input(shape=(NUMBER_OF_BATCHES, 30)) # This returns a tensor. Here we are feeding NUMBER_OF_BATCHES transactions at a time, each with the 30 features.

#2.- Define the Conv1D
conv1 = Conv1D(32, (5),           # 32 filters with a window of width 5
               strides=1,         # think autoregression
               padding='causal',  # forward in time
               dilation_rate=1,   # ignore this and everything that follows are default parameters
               activation='relu', 
               use_bias=True,
               kernel_initializer='glorot_uniform', 
               bias_initializer='zeros',
               kernel_regularizer=None, # no regularisation for the moment
               bias_regularizer=None, 
               activity_regularizer=None,
               kernel_constraint=None, 
               bias_constraint=None)(inputs) # syntax to chain layers: Layer(...)(PreviousLayer)


#3.- Add a fully connected layer. Define it to have 64 neurons after that and relu neurons (note that the choice of 64 is fairly arbitrary, we picked it to have something "large but not too large" but there's not much more than guesswork here as, unfortunately, with much of "deep learning"). Again, we chain that layer to the previous one.
fc1 = Dense(64, activation='relu')(conv1)

#4.- Define the output layer. For this, we need a softmax layer with 2 neurons (two classes: 0/1). 
predictions = Dense(2, activation='softmax')(fc1)

#5.- Wrap the model, mentioning the input and output layers
model = Model(inputs=inputs, 
              outputs=predictions)

#6.- Compile the model. Here we choose "rmsprop" to do the training but you could use Adam etc
# the loss is the standard loss for classification and we want to show the accuracy.
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#7.- Fit the model
model.fit(X_train_s_batch, y_train_batch, epochs=30) 




"""
############### (Vanilla) RNN (for sequences and time series modeling)
"""
#The RNN is globally better than the CNN for time series.
#Recurrent Neural Networks (RNNs) are often the model of choice for sequence data

######0.- Preprocess input data. 
#It goes similar to the CNN for time series data, 
pipeline = Pipeline([
    ('scaling', StandardScaler()),
])
preprocessor = pipeline.fit(X_train)
X_train_s = preprocessor.transform(X_train)
X_test_s = preprocessor.transform(X_test)

#And reshape to batches
BATCH_SIZE = 100
X_train_s_batch = reshape_to_batches(X_train_s, BATCH_SIZE)
#NOTE: that the batch size is particularly important because this is the sequence size that we are going to train the RNN on. This means that any dependencies further apart thanBATCH_SIZE will not be taken into account. We could in theory give only one batch with the entire sequence but that will take an excessive amount of time to train and success is not guaranteed (vanishing gradient problem).

#Re-encode and re-shape the labes
y_binary = to_categorical(y_train)
y_train_batch = reshape_to_batches(y_binary, BATCH_SIZE)

####Make the model

#1.- Define the input layer
inputs = Input(shape=(BATCH_SIZE, 30)) #this 30 value comes from X_train_s_batch.shape[2]

#2.- Define the SimpleRNN() layer. By default, Keras considers the the many-to-one architecture, sometimes also known as an encoder. However, we want to perform a prediction at every time step. Therefore, we make the RNN layer return output for every sequence with return_sequences=True

rnn = SimpleRNN(64,                                     #Number of neurons
                activation='tanh',                      #Use the tanh activation function
                use_bias=True, 
                kernel_initializer='glorot_uniform',    #the initializer is the Glorot intializer, centered at zero
                recurrent_initializer='orthogonal', bias_initializer='zeros', 
                kernel_regularizer=None, recurrent_regularizer=None, 
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 
                recurrent_constraint=None, bias_constraint=None, 
                dropout=0.0,                            #No dropout
                recurrent_dropout=0.0, 
                return_sequences=True, 
                return_state=False, go_backwards=False, stateful=False, unroll=False)(inputs)


#3.- Define the output layer
#We define it with 2 dimensions given that there are two classes (we're still in the classification context).
predictions = TimeDistributed(Dense(2, activation='softmax'))(rnn)

#4.- Define the model
rnn_model = Model(inputs=inputs, 
              outputs=predictions)

#5.- Compile the model
rnn_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#6.- Fit the model
rnn_model.fit(X_train_s_batch, y_train_batch, epochs=15)



"""
############### LSTM (A type of RNN, for sequences and time series classification)
"""
#Long-short term memory

#0.- Preprocess teh data like for the Vanilla RNN

#1.- Define the input layer
inputs = Input(shape=(BATCH_SIZE, 30)) #this 30 value comes from X_train_s_batch.shape[2]

#2.1- Define the first LSTM() layer
lstm1 = LSTM(64, 
            activation='tanh', 
            recurrent_activation='hard_sigmoid', 
            use_bias=True, 
            kernel_initializer='glorot_uniform', 
            recurrent_initializer='orthogonal', 
            bias_initializer='zeros', 
            unit_forget_bias=True, 
            kernel_regularizer=None, 
            recurrent_regularizer=None, 
            bias_regularizer=None, 
            activity_regularizer=None, 
            kernel_constraint=None, 
            recurrent_constraint=None, 
            bias_constraint=None, 
            dropout=0.0, 
            recurrent_dropout=0.0, 
            implementation=1,      #the implementation parameter determines whether your hardware is cpu (1) or gpu (2)
            return_sequences=True, 
            return_state=False, 
            go_backwards=False, 
            stateful=False,
            unroll=False)(inputs)


#If you want to use regularisation with l1_l2, do

#            kernel_regularizer=keras.regularizers.l1_l2(0.01), 
#            recurrent_regularizer=keras.regularizers.l1_l2(0.01),

#If you want to do regularization with dropout. when using this, you will need at least 25 epochs to get decent results.

#            dropout=0.2, 
#            recurrent_dropout=0.05,


#2.2.- Stack layers
lstm2 = LSTM(64, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, 
            kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
            bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
            recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
            kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, 
            recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, 
            go_backwards=False, stateful=False, unroll=False)(lstm1)


#3.- Define output layer. We give a 2 dimensional (given that we have 2 classes) softmax output layer (If you keep the model with only one layer, then the last value is lstm1, not lstm2)
predictions = TimeDistributed(Dense(2, activation='softmax'))(lstm2)

#4.- Define the model
lstm_model = Model(inputs=inputs, 
                   outputs=predictions)

#5.- Compile the model
lstm_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#6.- Fit the model
lstm_model.fit(X_train_s_batch, y_train_batch, epochs=15)

#7.- Run the model to make predictions
y_pred_lstm = lstm_model.predict(X_test_s_batch)

#8.- Evaluate, as shown below in te evaluation section




"""
############### LSTM (A type of RNN, for sequences and time series regression)
"""

#########################################
########For predicting only one feature
#########################################



########Preprocess data

#0.1.- Preprocess the data
df["Date(UTC)"] = pd.to_datetime(df["Date(UTC)"])
df = df.set_index("Date(UTC)")
#Check for the NaNs columns. Drop column(s) or impute as needed. 
#And scale the data (StandardScaler())

#0.2.- Make the vector of y's, considering that we want to predict the price of tomorrow
id_price_column = 1 # index of the appropriate column
y = np.expand_dims(X[1:, id_price_column], -1) #the first 1 here is because we left out of the prediction the first measure
#Also, initially we are going to exclude the price feature from the data set entirely, and also the last price measured
X_ = X[0:-1, np.arange(X.shape[1]) != id_price]

#0.3.- Train test split for time series data
def train_test_split_time_series_regres(X, y, test_ratio=0.15):
    total_samples = X.shape[0]
    train_idx = int(total_samples * (1-test_ratio))
    XTrain = X[:train_idx]
    yTrain = y[:train_idx]
    XTest = X[train_idx:]
    yTest = y[train_idx:]
    return XTrain, yTrain, XTest, yTest

XTrain, yTrain, XTest, yTest = train_test_split_time_series_regres(X_, y)

#0.4.- Apply the reshape_to_batches with batch size of 30
BATCH_SIZE = 30
XTrain_batch = reshape_to_batches(XTrain, BATCH_SIZE)
yTrain_batch = reshape_to_batches(yTrain, BATCH_SIZE)


#0.5.- you also need to reshape the test set
XTest_batch = reshape_to_batches(XTest, BATCH_SIZE)
yTest_batch = reshape_to_batches(yTest, BATCH_SIZE)


####Make teh model

#1.- Make the input layer. Because we would like to be able to predict the price given only yesterday's information, we will allow our network to accept batches of any size. This is accomplished by supplying None as a shape parameter.
inputs = Input(shape=(None, 16)) #this 16 is the XTest_batch.shape[2] value. This returns a tensor

#2.- Define the LSTM layer
lstm = LSTM(64, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, 
            kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
            bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
            recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
            kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, 
            recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, 
            go_backwards=False, stateful=False, unroll=False)(inputs)

#If training is very fast here, you can play around with the number of neurons and number of layers to try to get better results. One way to go is to increase the number of neurons (say to 256 instead of 64); feel free to tweak.
#Or you could try something more complicated, for example:
#two layers
#some dropout
#some regularisation
#more epochs

#3.- Define output layer. The main difference with classification is that the activation of the last layer is linear
predictions = TimeDistributed(Dense(1, activation='linear'))(lstm)

#4.- Define the model
lstm_model = Model(inputs=inputs, outputs=predictions)

#5.- Compile. The second difference is the loss function (now MSE)
lstm_model.compile(optimizer='rmsprop',
                   loss='mean_squared_error',
                   metrics=['accuracy'])

#6.- Fit
lstm_model.fit(XTrain_batch, yTrain_batch, epochs=100)

#7.- Generate the predictions
y_pred_lstm = lstm_model.predict(XTest_batch)
#These values don't really tell you much at this point. Remember that you have normalised the values with 0-mean and variance 1 so that the small numbers should not be particularly impressive.



#########################################
########Generator: For predicting multiple features
#########################################



########Preprocess data

#0.1.- Preprocess the data
df["Date(UTC)"] = pd.to_datetime(df["Date(UTC)"])
df = df.set_index("Date(UTC)")
#Check for the NaNs columns. Drop column(s) or impute as needed. 
#And scale the data (StandardScaler())

#0.2.- Make the shifted vectors
y_shift = X[1:, :] # the "future"
X_shift = X[0:-1, :] # the "past"

#0.3.- Train test split for time series data
def train_test_split_time_series_regres(X, y, test_ratio=0.15):
    total_samples = X.shape[0]
    train_idx = int(total_samples * (1-test_ratio))
    XTrain = X[:train_idx]
    yTrain = y[:train_idx]
    XTest = X[train_idx:]
    yTest = y[train_idx:]
    return XTrain, yTrain, XTest, yTest

XTrain, yTrain, XTest, yTest = train_test_split_time_series_regres(X_, y)

#0.4.- Apply the reshape_to_batches with batch size of 30
BATCH_SIZE = 30
XTrain_batch = reshape_to_batches(XTrain, BATCH_SIZE)
yTrain_batch = reshape_to_batches(yTrain, BATCH_SIZE)


#0.5.- you also need to reshape the test set
XTest_batch = reshape_to_batches(XTest, BATCH_SIZE)
yTest_batch = reshape_to_batches(yTest, BATCH_SIZE)


####Make the model

#1.- Make the input layer.
inputs = Input(shape=(None, 17)) # #this 17 is the XTest_batch.shape[2] value. This returns a tensor

#2.- Define first layer
lstm = LSTM(512, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, 
            kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
            bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
            recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
            kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, 
            recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, 
            go_backwards=False, stateful=False, unroll=False)(inputs)


#3.- Define output model
predictions = TimeDistributed(Dense(17, activation='linear'))(lstm)

#4.- Define model
model_lstm512 = Model(inputs=inputs, outputs=predictions)

#5.- Compile
model_lstm512.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

#6.- Fit
model_lstm512.fit(X_batch, y_batch, epochs=500)

#7.- Predict. In order to make our network tell the future, we need to make it generate the values for tomorrow and then iteratively feed this back into the network.
#remember that we left out the last X sample, we can start from there
days = 3 * 30
X_last = X[-1, :] #X_last = X[:, :] if you want to give it the entire history, in which case the NN is trying to learn the joint probability of the features.

X_batch = np.swapaxes(np.expand_dims(np.expand_dims(X_last, -1), -1), 0, 2)
#X_batch = np.swapaxes(np.swapaxes(np.expand_dims(X_last, -1), 0, 2), 1, 2) #if you want to give it the entire history. 

for day in range(days):
    print("Day #{} - {} data points".format(day, X_batch.shape[1]))
    y_pred = model_lstm512.predict(X_batch)
    # we are only going to use the most recent prediction
    # otherwise the prediction power could quickly deteriorate
    y_pred = np.swapaxes(np.expand_dims(np.expand_dims(y_pred[0, X_batch.shape[1]-1, :], -1), -1), 0, 2)
    X_batch = np.concatenate([X_batch, y_pred], axis=1)

df = pd.DataFrame(
        pipeline.inverse_transform(
            X_batch.reshape(X_batch.shape[1], X_batch.shape[2])))
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], unit='s')




"""
############### Bi-directional RNN (A type of RNN for sequences and time series classification)
"""

#A RNN can be run simultaneously from "both directions":
#   one "forward" in time
#   one "backward in time
#So, obviously, this is better for sequences data than for time-series data, because in time-series data we are interested in predicting the future, not in redicting what we already know that happened. anyway, it can be aplied to time-series data too. 



#0.- Preprocess teh data like for the Vanilla RNN

#1.- Define the input layer
inputs = Input(shape=(BATCH_SIZE, 30)) #this 30 value comes from X_train_s_batch.shape[2]

#2.1.- Define the LSTM() forward pass.
lstm_fwd = LSTM(64, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, 
            kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
            bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
            recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
            kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, 
            recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, 
            ### GO FORWARD
            go_backwards=False, stateful=False, unroll=False)(inputs)


#2.2.- Define the LSTM() backward pass. Note that you give the same input (inputs) to the backward LSTM as those to the forward
lstm_bck = LSTM(64, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, 
            kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
            bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
            recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
            kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, 
            recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, 
            ### GO BACKWARD
            go_backwards=True, stateful=False, unroll=False)(inputs)

#3.- Merge the results of the two layers
merge = Concatenate(axis=-1)([lstm_fwd, lstm_bck])

#4.- Define output layer
predictions = TimeDistributed(Dense(2, activation='softmax'))(merge)

#5.- Define model
bidir_model = Model(inputs=inputs, 
                    outputs=predictions)

#6.- Compile
bidir_model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

#7.- Fit
bidir_model.fit(X_train_s_batch, y_train_batch, epochs=15)

#8.- Predict
y_pred_bidir = bidir_model.predict(X_test_s_batch)





"""
############### GRU (A type of RNN for sequences and time series classification)
"""
#Gated Recurrent Unit


#0.- Preprocess teh data like for the Vanilla RNN

#1.- Define the input layer
inputs = Input(shape=(BATCH_SIZE, 30)) #this 30 value comes from X_train_s_batch.shape[2]

#2.- Define the GRU() layer
gru = GRU(64, 
          activation='tanh', 
          recurrent_activation='hard_sigmoid',
          use_bias=True, 
          kernel_initializer='glorot_uniform',
          recurrent_initializer='orthogonal', 
          bias_initializer='zeros',
          kernel_regularizer=None, 
          recurrent_regularizer=None, 
          bias_regularizer=None,
          activity_regularizer=None, 
          kernel_constraint=None, 
          recurrent_constraint=None,
          bias_constraint=None, 
          dropout=0.0, 
          recurrent_dropout=0.0, 
          implementation=1,
          return_sequences=True, 
          return_state=False, 
          go_backwards=False, 
          stateful=False, 
          unroll=False)(inputs)

# 3.- Define output layer. (We use 2 because that's the number of categories in this example dataset)
predictions = TimeDistributed(Dense(2, activation='softmax'))(gru)

#4.- Model definition
gru_model = Model(inputs=inputs, outputs=predictions)

#5.- Compilation 
gru_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#6.- Fitting
gru_model.fit(X_train_s_batch, y_train_batch, epochs=15)

#7.- Run the model for predictoin
y_pred_gru = gru_model.predict(X_test_s_batch)


#8.- Evaluate, as shown below in te evaluation section




""""
##############################################
########################################## NLP 
##############################################
""""

#Natural Language Processing


"""
############### Bag of words (BoW) for binomial classification
"""

#NLP for supervised learning (binary classification in this case) with bag of words, can be used e.g. for sentiment analysis.

#0.- First, split the text data
all_reviews = df.sample(frac=1).reset_index(drop=True) #Shuffle the entries if you have them all first the category 1 and then first the category 2.
              
train_data, train_targets, test_data, test_targets = train_test_split(all_reviews.text, target_reviews.label, test_size=0.2)

###Preprocess input text

#1.- Tokenize, lemmatize, make lower case, and remove stop words and punctuation marks.
              
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
import string
punctuations = string.punctuation

nlp = spacy.load('en_core_web_lg')

def spacy_tokenizer(text):
    tokens = nlp(text)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return (' '.join(tokens))

#1.2.- Apply the function to train and test datasets
train_data = [spacy_tokenizer(text) for text in train_data]
test_data = [spacy_tokenizer(text) for text in test_data]


#2.- If the target vector is a string, turn it into numerical values. E.g. 1 for positive entries, 0 for negative entries
def fix_target(target_set)
    for label in target_set:
        target_fixed = []
        if label=="pos": target_fixed.append(1)
        else: target_fixed.append(0)
    return target_fixed

train_targets = fix_target(train_targets)
test_targets = fix_target(test_targets)


#3.- Extracting the features. Calculate word frequency. Note that this is fit only on the training data
              
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data)
#This gives a matrix, made up of a vector of the freq of all words present in all the docs in the dataset for each document of the dataset. The documents are the rows, and the words are teh columns. E.i. number of docs  ×  number of words in the shared vocabulary.
count_vect.get_feature_names()[5729] #To see the word of a particular word index
              

#4.- Normalize for the term frequency and the inverse document frequency. Note that this is fit only on the training data

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


#5.- Training a classifier for Binary classification with naïve Bayes. Note that this is fit only on the training data
              
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf, train_targets)

#6.- Predict on the test dataset
X_test_counts = count_vect.transform(test_data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
predicted = clf.predict(X_test_tfidf)

#6.2.- You can condence points 3-6 into a pipeline:

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())   
                     #To use LSVC instead of MNB, use from sklearn.svm import LinearSVC; LinearSVC()
                     #To use SGDClassifier, use from sklearn.linear_model import SGDClassifier; SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=50, tol=1e-3, random_state=42,)

                    ])

text_clf.fit(train_data, train_targets)
predicted = text_clf.predict(test_data)   

#6.3.- To tune the parameters of the pipeline, do a grid search
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
              'clf__penalty': ('l1', 'l2')}
gs_clf = GridSearchCV(text_clf, parameters)
gs_clf = gs_clf.fit(train_data, train_targets)
predicted = gs_clf.predict(test_data)


#7.- Evaluate the classifier
from sklearn.metrics import accuracy_score
print(accuracy_score(test_targets, predicted))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_targets, predicted))

from sklearn.metrics import classification_report
print(classification_report(test_targets, predicted))


"""
###############Multi-class tasks
"""              
            
#Follow the first steps from above (steps 0 and 1 and 1.2), then:
              
def text2vec(train_set, test_set):
    vectorizer = TfidfVectorizer(stop_words = 'english')
    vectors_train = vectorizer.fit_transform(train_set.data)
    vectors_test = vectorizer.transform(test_set.data)
    return (vectorizer, vectors_train, vectors_test)
              
categories = ['talk.politics.misc', 'sci.electronics', 'comp.sys.mac.hardware', 'rec.autos']
           
vectorizer, vectors_train, vectors_test = text2vec(train_data, test_data)


clf=MultinomialNB(alpha=.01)

def classify(vectors_train, vectors_test, train_set, clf):
    clf.fit(vectors_train, train_set.target)
    predictions = clf.predict(vectors_test)
    return predictions


predictions = classify(vectors_train, vectors_test, 
                       train_data, clf)
full_report = classification_report(test_data.target, predictions, 
                                    target_names=test_data.target_names)
print(full_report)


#Get Most predictive features
              
def show_top10(classifier, categories, vectorizer):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))


show_top10(clf, categories, vectorizer)

              



""""
##############################################
########################## Recommender systems
##############################################
""""


#0.- Some EDA. For example:
# We may need to assign a unique number between (0, #users) to each user 
# and a unique number between (0, #movies) to each movie
# Assign these numbers to columns "userId" and "movieId" in the ratings dataframe 
# Screen the first 5 rows of the ratings dataframe 

df['userId']  = df.userId.astype('category').cat.codes.values
df['movieId'] = df.movieId.astype('category').cat.codes.values



"""
###############Non-personalised recommenders - Popularity model
"""

#Despite their simplicity, popularity models are often used as baseline models or to overcome cold-start cases. 
popular_movies = ratings_full.groupby(['title'])['userId'].count().\
                              sort_values(ascending=False).\
                              reset_index().head()




"""
###############Content-based recommenders - NLP basis
"""

#1.- Get the TF-IDF matrix:
#TF-IDF will return a matrix where each column represents a word in the vocabulary and each row represents a movie.)

tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
movie_input = movies_df['title'] + movies['genres']
tfidf_matrix = tfidf.fit_transform(movie_input)
feature_names = tfidf.get_feature_names()

tfidf_df = pd.DataFrame(tfidf_matrix.todense(), index=movies['title'], columns=feature_names)

#2.- Compute the pairwise title similarities
#With the tfidf matrix you can now compute the pairwise similarity scores.

cosine_similarities = cosine_similarity(tfidf_matrix)

#3.- Make the recommendations

def get_recommendations(title, movies_df, cosine_similarities): 
    # Get the index of the movie that matches the given title
    idx = movies_df[movies_df['title']==title].index.item()
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_similarities[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 5 most similar movies
    sim_scores = sim_scores[1:6]
    # Get the movie indices
    movie_indices  = [i[0] for i in sim_scores]
    similar_movies = df.loc[movie_indices,['title','genres','tag']]
    return similar_movies

get_recommendations("Toy Story", movies_df, cosine_similarities)


"""
############### Collaborative Filtering recommenders - Memory-based
"""

#Collaborative filtering (CF) is considered the "workhorse" of recommender systems and is widely-used on shopping websites like Amazon, news services, and video content providers like YouTube. The key idea behind CF is that similar users share the same interest and that similar items are liked by a user. CF can be divided into Memory-Based Collaborative Filtering and Model-Based Collaborative filtering.

#1.- Train/test split the data
#The main difference in the case of the recommender system is that, even though the same preferences (ratings) are not present in both train and test, the dimensionality of the two datasets needs to be equal to n users x m items (in terms of unique users and unique items, so, like print(train_data['userId'].nunique()); print(train_data['movieId'].nunique()))

train_data, test_data = cv.train_test_split(ratings_full,
                                            test_size=0.20, 
                                            stratify=ratings_full['userId'],
                                            random_state=0)



#2.- Construct the interaction matrix

#Most recommendation models consist of building a user-by-item matrix with some sort of "interaction" number in each cell. Assume there are m users and n items, we tend to use a matrix with size m x n to denote the past behaviour of users. Most recommender models consist of building a user-by-item (utility) matrix with some "interaction" number in each cell. For instance, M_{i, j}_ denotes how user i likes item j. Such matrix is called utility matrix.

#The purpose of CF is to fill in the blanks (cells) in the utility matrix that a user has not seen/rated before based on the similarity between users or items.

## Create the utility matrices (one for train and one for test)
#The dimensionality of train and test utility matrices need to be the same.

train_ratings = np.zeros((n_users, n_items))
test_ratings  = np.zeros((n_users, n_items))

for row in train_data.itertuples():
    train_ratings[row.userId, row.movieId] = row.rating
    
for row in test_data.itertuples():
    test_ratings[row.userId, row.movieId] = row.rating


#3.- Get the similarities 

#Memory-based algorithms are easy to implement and produce reasonable prediction quality. The drawback of memory-based CF is that it doesn't scale to real-world scenarios and doesn't address the well-known cold-start problem, that is when new user or new item enters the system. 
#Model-based CF methods are scalable and can deal with higher sparsity level than memory-based models, but also suffer when new users or items that don't have any ratings enter the system.

#There are two categories of memory-based CF:

#   User-based CF: measure the similarity between target users and other users
#   Item-based CF: measure the similarity between the items that target users rates/ interacts with and other items


from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

# Calculate the user similarity with the pairwise_distances() 
# function on the train data

user_similarity = pairwise_distances(train_ratings, metric='cosine')
item_similarity = pairwise_distances(train_ratings.T, metric='cosine')

#4.- Make the predictions

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff)
        / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

user_prediction = predict(train_ratings, user_similarity, type='user')
item_prediction = predict(train_ratings, item_similarity, type='item')

#user_prediction and item_prediction must have the same shape
              

"""
############### Collaborative Filtering recommenders - LightFM
"""

#LightFM is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback.

#It also makes it possible to incorporate both item and user metadata into the traditional matrix factorization algorithms. It represents each user and item as the sum of the latent representations of their features, thus allowing recommendations to generalise to new items (via item features) and to new users (via user features).

#The model learns embeddings (latent representations in a high-dimensional space) for users and items in a way that encodes user preferences over items. When multiplied together, these representations produce scores for every item for a given user; items scored highly are more likely to be interesting to the user.

#LightFM is a traditional collaborative filtering matrix factorization method.

#0.- Set the parametrs up

from lightfm import LightFM
from lightfm.evaluation import recall_at_k, precision_at_k, auc_score, reciprocal_rank

# Set some initial hyperparameter values
NUM_THREADS = 2 
NUM_COMPONENTS = 30
EPOCHS = 30
LEARNING_RATE=0.05

#1.- Convert to sparse matrices

train_sparse = sparse.csr_matrix(train_ratings)
test_sparse = sparse.csr_matrix(test_ratings)

#2.- Initiate and fit the model
#Using as loss function the Weighted Approximate-Rank Pairwise (WARP), which helps us create recommendations in a hybrid way (content-based+collaborative=hybrid)

cf_model = LightFM(loss='warp',
                   learning_rate=LEARNING_RATE,
                   no_components=NUM_COMPONENTS,
                   random_state=0)

cf_model = cf_model.fit(train_sparse, epochs=EPOCHS)

#3.- Make predictions

def recommendations(model, user_id, rec_num=10):
    # movies the user has already rated
    rated_movies = ratings_full[ratings_full['userId']==user_id]
    # movies the RecSys model predicts they will like
    scores = model.predict(user_id, np.arange(n_items))
    # Rank them in order of most to least liked
    top_items = movies.iloc[np.argsort(-scores)]
    top_items = top_items[~(top_items.index).isin(rated_movies['movieId'])]
    top_items = top_items[['title','genres']].head(rec_num)
    top_items.reset_index(inplace=True, drop=True)
    return top_items, rated_movies[['title','rating','genres']]


top_items, rated_movies = recommendations(cf_model, user_id = 0)


#4.- Visualize user and items embeddings, to see if there's any clustering pattern

#Normalize the embeddings
item_embeddings = (cf_model.item_embeddings.T
                      / np.linalg.norm(cf_model.item_embeddings, axis=1)).T

user_embeddings = (cf_model.user_embeddings.T
                      / np.linalg.norm(cf_model.user_embeddings, axis=1)).T


#Visualize the embeddings with PCA and TSNE
              
pca = PCA(n_components=2)
pca_items = pca.fit_transform(item_embeddings)
# Visualise the PC scores
sns.scatterplot(x=pca_items[:,0], y=pca_items[:,1], s=7)
plt.show()

tsne = TSNE(n_components=2, random_state=0)
tnse_items = tsne.fit_transform(item_embeddings)
# Visualise the TSNE scores
sns.scatterplot(x=tnse_items[:,0], y=tnse_items[:,1], s=7)
plt.show()


"""
############### Hybrid recommenders - LightFM
"""

ITEM_ALPHA = 1e-6

hybrid = LightFM(loss='warp', 
                 learning_rate=LEARNING_RATE,
                 item_alpha=ITEM_ALPHA, 
                 no_components=NUM_COMPONENTS,
                 random_state=0)


hybrid = hybrid.fit(train_sparse,
                    item_features=tfidf_matrix,
                    epochs=EPOCHS, 
                    num_threads=NUM_THREADS)

top_items, rated_movies = recommendations(hybrid, user_id = 50)
top_items



""""
##########################################################################
########################### THIRD PHASE #################################
##########################################################################
""""


""""
##############################################
####################### EVALUATION METRICS 
##############################################
""""

"""
###############Unsupervised
"""

silhouette_score(df, cluster_assignment)

#To evaluate the clustering algorithm, you can make use of the following measures:
#   Homogeneity – a measure that evaluates if each of the clusters contains only data points from a single class (note the similarity to precision for classification algorithms).
#   Completeness – a measure that tells if all the data points from a particular class are elements of the same cluster (note the similarity to recall for classification algorithms).
#   V-measure - You may note that homogeneity and completeness are complementary to each other. V-measure is the harmonic mean between the two that allows to take both into account (note the similarity to  𝐹1  _score_: 𝑣=2×ℎ𝑜𝑚𝑜𝑔𝑒𝑛𝑒𝑖𝑡𝑦×𝑐𝑜𝑚𝑝𝑙𝑒𝑡𝑒𝑛𝑒𝑠𝑠ℎ𝑜𝑚𝑜𝑔𝑒𝑛𝑒𝑖𝑡𝑦+𝑐𝑜𝑚𝑝𝑙𝑒𝑡𝑒𝑛𝑒𝑠𝑠



"""
###############Supervised - Discrete classifiers
"""

#Get the accuracy of your predictions
accuracy_score(y_test, y_pred)

#Get the confusion matrics ([[TP, FN], [FP, TN]])
cm = confusion_matrix(y_test, y_pred)
#And plot it
sns.heatmap(cm, annot=True, fmt='d')
#Normalized by row
sns.heatmap(cm/cm.sum(axis=1, keepdims=True), annot=True, fmt='.3f')

#Get the precision, recall and f1 score too
#precision = nr_true_positives / nr_predicted_positive
#recall = sensitivity = TPR = nr_true_positives / nr_positive_cases
#f1 = 2*(precision*recall) / (precision+recall)
classification_report(y_test, y_pred, digits=3, output_dict=True)
print(classification_report(y_test, y_pred, target_names=le.classes_))  #For when you have multiple classes, because the sklearn prints the classed targets, not the names

precision = classification_report(y_train, y_pred, digits=3, output_dict=True)['1.0']['precision']
recall = classification_report(y_train, y_pred, digits=3, output_dict=True)['1.0']['recall']
f1_score = classification_report(y_train, y_pred, digits=3, output_dict=True)['1.0']['f1-score']


precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)
roc_auc = metrics.roc_auc_score(y_test, y_pred)



#####With cross-validation
cv_resuts= cross_validate(model, X_train, y_train, return_train_score=True, cv=10, n_jobs=4)
cv_results['train_score'].mean()
cv_results['test_score'].mean()




"""
###############Supervised - Probabilistic classification
"""

y_pred_probs = clf.predict_proba(X_test) #just use .predict_proba instead of .predict

#The Bernoulli log likelihood loss, (also known as cross entropy loss)
log-loss= metrics.log_loss(y_test, y_pred_probs)
#The Brier score (also known as Mean Squared Error, MSE)
brier =  metrics.brier_score_loss(y_test, y_pred_probs)
#The primary difference between the two is how they are bounded. The Brier-score is 0-1 bounded, with 0 for perfect predictions, and 1 for predictions which are completly wrong. However, the log-loss is 0-infinite bounded, again is 0 for perfect predictions, and is infinite if a single prediction is completely wrong.
#This is due to how we treat incorrect 'certain' classifications, i.e. predicting a 0 for an event that occured or a 1 for an event which did not. In these cases the log-likelihood becomes infinite (as  ln(0)  is infinite), whereas the BS score only increases by  1𝑁 .
#The log-loss has a particular advantage when training and optimising a model - minimising the log-loss is equivalent to maximising the probability of the data given the model, and as such is know as Maximum Likelihood Estimation (MLE)



#Get your ROC curve: The ROC curve plots the TPR vs FPR (given a threshold to classify the predicted probability as being True or False).
fpr, tpr, thresh = roc_curve(y_test, y_pred_probs)
plt.plot(fpr, tpr, 'o-')

#Get the AUC (area under the curve). For AUC, the larger the better.
roc_auc_score(y_test, y_pred_probs)



"""
###############Time series
"""
#########A metric to evaluate autoregression models is
mean_absolute_error(time_series[optlag:], prediction)



"""
###############Time series CNN/RNN model
"""
############ Evaluate the model time series CNN/RNN model results

#1.- first transform the test data into the appropriate shape
X_test_s_batch = reshape_to_batches(X_test_s, NUMBER_OF_BATCHES)
#2.- And make the y categorical
y_binary = to_categorical(y_test)
#3.- and reshape the categorized y
y_test_batch = reshape_to_batches(y_binary, 100)
#4.- make the prediction with the trained model
y_pred = model.predict(X_test_s_batch)
#5.- store the raw predictions we will need them in a bit
y_hat = np.copy(y_pred)
#6.- Make a function to reshape the preds in order to use on them the sklearn evaluation metrics, which use as input a single vector instead of a tensor
def convert_3d_to_2d(arr):
    return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])

#7.- Sklearn metrics functions expect a single vector, containing either a probability score or a confidence interval. Further, since our labels are binary labels, we can only compare them if our results are also binary. Hence, we are going to make a simplifying assumption: all classifications where there is a higher than 50% chance for a given class are going to be assigned that class and vice versa.
threshold = 0.5
y_pred[np.where(y_pred >= threshold)] = 1
y_pred[np.where(y_pred < threshold)]  = 0
print(confusion_matrix(
        convert_3d_to_2d(y_test_batch)[:, 1], 
        convert_3d_to_2d(y_pred)[:, 1]))
print(classification_report(
        convert_3d_to_2d(y_test_batch)[:, 1], 
        convert_3d_to_2d(y_pred)[:, 1],
        target_names = ["Genuine", "Fraud"],
        digits = 5))

#8.- Use a ROC curve to find what is an ideal pair TPR/FPR
# long way, allows to plot the curve
fpr, tpr, thresholds = roc_curve(convert_3d_to_2d(y_test_batch)[:, 1], 
                                 convert_3d_to_2d(y_hat)[:, 1])
print(auc(fpr, tpr))

#With this you can find exactly what is the threshold you need to use in order to get X FDR
#For example, if you want 0.6% FDR, what's the threshold to use:
# find the value of FPR, where it is 0.6% or slightly above
fpr_id = np.where(fpr >= 0.006)[0][0]
print("FPR {:2.2f}%".format(fpr[fpr_id]*100))
print("TPR {:2.2f}%".format(tpr[fpr_id]*100))
print("threshold {:.2e}".format(thresholds[fpr_id]*100))

#If adding more layers and having many neurons (many parameters) , 
#there are two ways to go about possible overfitting in the hope that a more 
#complex model might lead to better performances (which is not necessarily true):
#1. decrease the number of parameters
#2. introduce regularisation
#So, you could start by reducing the number of parameters `64-->32`, 
#and do exactly the same as before but with only 32 neurons per layer. 
#Regularization also helps. With keras it is particularly easy to add any form of regularisation you want, either using the

# [component]_regularizer parameter (penalise components that are too far from sensible values) or
# the [component]_constraint parameter (clip components to be within a set range).

#In the first case, you can apply both l1 and l2 of the regularisation techniques you have learned so far regulariser docs. You can also add constraints (min norm, max norm, etc see cons_train docs)

#Of course, picking the parameters of the regularisation is hard and there is no good simple generic technique to do it. You could think about CV but here it would just be computationally too expensive. There are some rule of thumbs in terms of what is "big" and what is "small" but none are really justified. This is where resources can make all the difference. If you have access to a bunch of GPUs (or better, TPUs) training one neural net with a set of regularisation parameters can be done in a reasonable time and therefore you could do a form of randomised CV. If you're on a single CPU on your laptop however, you probably should not attempt doing hyperparameter tuning, your time is probably best invested buying credits off a cloud computing provider and using their GPUs paying per hour of use.
#If regularization gives worst metric values, it could be either that we applied an unreasonably high regularisation value or that it is much harder to optimise the problem with regularisation and the optimisation algorithm needs more epochs...

#One of the most effective forms of regularisations in the context of Neural Networks is Dropout. There are two places where we can use dropout:

#on the input connection
#on the reccurent connections.

#a dropout on the connection means that the data on that connection to each LSTM cell will be excluded from node activation and weight updates with a given probability. The dropout value is a percentage between 0 (no dropout) and 1 (no connection).

#Just remember that regularisation is difficult to tune, requires a lot of practice and it doesn't hurt to have large computational resources...



"""
###############Recommender systems
"""

#In Recommender Systems, there are a set metrics commonly used for evaluation. We chose to work with Top-N accuracy metrics, which evaluate the accuracy of the top recommendations provided to a user, comparing to the items the user has actually interacted in test set.
              
from lightfm.evaluation import recall_at_k, precision_at_k, auc_score, reciprocal_rank

train_precision = precision_at_k(cf_model, 
                                 train_sparse, 
                                 k=10).mean()

test_precision  = precision_at_k(cf_model, 
                                 test_interactions=test_sparse, 
                                 train_interactions=train_sparse, 
                                 k=10).mean()

#Measure the recall at k metric for a model: the number of positive items in the first k positions of the ranked list of results divided by the number of positive items in the test period. A perfect score is 1.0. The function returns a numpy array containing recall@k scores for each user. If there are no interactions for a given user having items in the test period, the returned recall will be 0.

train_recall = recall_at_k(cf_model, 
                           train_sparse, 
                           k=10).mean()

test_recall  = recall_at_k(cf_model, 
                           test_interactions=test_sparse,
                           train_interactions=train_sparse,  
                           k=10).mean()


#Measure the ROC AUC metric for a model: the probability that a randomly chosen positive example has a higher score than a randomly chosen negative example. A perfect score is 1.0. The function returns a numpy array containing AUC scores for each user. If there are no interactions for a given user the returned AUC will be 0.5.

train_auc = auc_score(cf_model, 
                      train_sparse).mean()

test_auc  = auc_score(cf_model, 
                      test_interactions=test_sparse,
                      train_interactions=train_sparse).mean() 

#Measure the reciprocal rank metric for a model: 1 / the rank of the highest ranked positive example. A perfect score is 1.0. The function returns a numpy array containing reciprocal rank scores for each user. If there are no interactions for a given user the returned value will be 0.0.

train_reciprocal_rank = reciprocal_rank(cf_model, 
                                        train_sparse).mean()

test_reciprocal_rank  = reciprocal_rank(cf_model, 
                                        test_interactions=test_sparse,
                                        train_interactions=train_sparse,).mean()




""""
##############################################
####################### MODEL INTERPRETABILITY
##############################################
""""

########Explore the importance of each feature

features = X.columns
importances = clf.feature_importances_  #Or if teh clf is from a gridSearc, clf.best_estimator_.feature_importances_ 
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()



"""
###############Eli5
"""

#Using Eli5 to get importance of features globally and locally

import eli5

#################For a Linear Regression model

eli5.show_weights(lr_model, feature_names=all_features)
#all_features is the colnames of the df

#This table gives us the weight associated to each feature. The amplitude tells us how much of an impact a feature has on the predictions on average, the sign tells us in which direction.
              
#We can also use eli5 to explain a specific prediction, let's pick a row in the test data:

i = 4
entryOfInterest = X_test.iloc[i]
entryOfInterest_pred = y_test.iloc[i]
              
eli5.show_prediction(lr_model, entryOfInterest,
                     feature_names=all_features, show_feature_values=True)



#################For a Decision Tree, Random Forest, or LightGBM model

eli5.show_weights(dt_model, feature_names=all_features)

#For this model, the most important feature seems to be pdays but we don't know if the more days the more likely it is that someone will subscribe, or the opposite. It can be useful to debug your model and know if it seems to pick up something that it shouldn't, but appart for that, it isn't too useful: you won't be able to properly explain what your model does to someone thanks to that.

#So, eli5's show_weights method is good, but for more complex models, such as trees the information provided starts to be less helpful. Since show_weights is accessing the internal weights of a model, it does not work with all algorithms, making it harder to compare different models you might have built.

#eli5 implements another technique called Permutation Importance that is model agnostic and works for any black box model. By shuffling at random the values of a feature, we can observe how that affects the predictions and quantify how important that feature is. If we repeat on all features, we can get the overall importance of each feature and compare them. Let's try to do that on our models.

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(clf_model, scoring="balanced_accuracy")

perm.fit(X_test, y_test)
eli5.show_weights(perm, feature_names=all_features)

#here the feature importance is only given as an amplitude, we do not know in what direction it impacts the outcome. But the interpretation is quite interesting

"""
###############LIME - Tabular data
"""

#LIME to generate local intepretations of black box models

#LIME stands for Local Interpretable Model-Agnostic Explanations. We can use it with any model we've built in order to explain why it took a specific decision for a given observation.

from lime.lime_tabular import LimeTabularExplainer

#NOTE: We need to make sure we use the training set without one hot encoding - X_train

#Lime needs the dataset that is passed to have categorical values converted to integer labels that maps to the values in categorical_names. For instance, label 0 for the column 2 will map to divorced. We will use a custom helper function to do so, that converts data from original to LIME and from LIME to original format.

from helpers import convert_to_lime_format #It's in the folder of this ADS class (ADS10)

categorical_names = {}
cat_values = preprocessor.named_transformers_["categorical"].categories_
for col, val in zip(cat_features, cat_values):
    categorical_names[df.columns.get_loc(col)] = list(val)

X_train_lime = convert_to_lime_format(X_train, categorical_names).values

#Build the explainer

explainer = LimeTabularExplainer(X_train_lime,
                                 mode="classification",
                                 feature_names=X_train.columns,
                                 categorical_names=categorical_names,
                                 categorical_features=categorical_names.keys(),
                                 random_state=42)


#Explain an observation of interest

i = 4
entryOfInterest = X_test.iloc[[i], :] #It's the same as X_test.iloc[[i]]
entryOfInterest_pred = y_test.iloc[i]

observation = convert_to_lime_format(entryOfInterest,categorical_names).values[0]
#y_pred_probs is the results obtained with model.predict_proba(X_train)

explanation = explainer.explain_instance(observation, y_pred_probs, num_features=5)
explanation.show_in_notebook(show_table=True, show_all=False)

#LIME is fitting a linear model on a local perturbated dataset. You can access the coefficients, the intercept and the R squared of the linear model by calling respectively .local_exp, .intercept and .score on your explanation.

print(explanation.local_exp)
print(explanation.intercept)
print(explanation.score)
#If your R-squared is low, the linear model that LIME fitted isn't a great approximation to your model, which means you should not rely too much on the explanation it provides

#You can sabe the explanation in an html
explanation.save_to_file("explanation.html")



"""
###############LIME - Image data
"""
              
#Lime is quite slow with images, so it's wiser to stick to a "shallow" deep learning model.

from lime.lime_image import LimeImageExplainer

explainer = LimeImageExplainer()
              
explanation = explainer.explain_instance(image[0], model.predict, 
                                         top_labels=2, num_samples=1000,
                                         random_seed=42)

from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

temp, mask = explanation.get_image_and_mask(classLabelOfInterest, positive_only=True, num_features=5, 
                                            hide_rest=True)
# plot image and mask together
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))


"""
###############SHAP
"""

######################SHAP to interpret LOCAL predictions

import shap
# Need to load JS vis in the notebook
shap.initjs()

explainer = shap.TreeExplainer(lgb_model)

i = 4
entryOfInterest = X_test.iloc[i]
entryOfInterest_pred = y_test.iloc[i]

shap_values = explainer.shap_values(observation)
#SHAP values are a way to know the contribution of each specific attribute of the observation in the final prediction
#`shap_values` is a list, check the len of the list and dimension of the first numpy array:
print(len(shap_values))
print(shap_values[0].shape)
#For binary classification, shap_values returns two numpy arrays, one for each output class. Each array contains the SHAP value associated to each of the features. For our given observation, SHAP value per feature can be positive or negative, and if we sum them we get the difference between the value our model predicted and its "expected value".
              
#If you use the model agnostic KernelExplainer, those values will be expressed as probabilities. With the TreeExplainer though, the unit will be specific to the library/model you are using. For lightbm's classifier that is log_odds.

# If you need to check in what unit the output is (probs or log_odds), run the following:
explainer.model.tree_output

explainer.expected_value
              
#we can compute the difference between our model's prediction for this specific observation

def average_log_odds(predictions):
    return np.mean(np.log(predictions / (1 - predictions)))
              
average_log_odds(lgb_model.predict_proba(observation)[:, 1]) - explainer.expected_value[1]

#shap implements a force_plot function that allows to nicely plot the shap values for our observation, call force_plot passing

shap.force_plot(explainer.expected_value[1], shap_values[1], features=observation, link="logit")
#This plot makes it easier to see the impact of each feature.


######################SHAP for global interpretation of a model
              
#If we compute SHAP values for multiple observations at a time, shap is able to plot them with force_plot in a similar way as above, but for all at once

#Let's plot 1000 random observations
observations = X_test_processed.sample(1000, random_state=42)

shap_values = explainer.shap_values(observations)
shap.force_plot(explainer.expected_value[1], shap_values[1], features=observations, link="logit")
#We can see our 1000 samples on the x-axis. The y-axis corresponds to the same scale we were looking at before, where blue values corresponds to the probability decreasing, red increasing.

shap.summary_plot(shap_values[1], features=observations)
#Here, each point corresponds to a SHAP value for a given feature for a given samples. Features are grouped on the y-axis and the x-axis represent the SHAP value, either negative or positive. The colour correspond to the value of the feature, either high or low.

#SHAP also implements a way to plot SHAP values as interaction between two features.
shap.dependence_plot("age", shap_values[1], observations)

#SHAP is really powerful and allows to understand your model both locally and globally.

#However, SHAP can get extremely slow to compute SHAP values using the model-agnostic KernelExplainer. That means that if your model is tree based (tree explainer), neural networks (deep explainer) or linear, you'll be able to get the best out of shap with good performance. But for other models, such as KNN, SVM or even a custom sklearn Pipeline object allowing you to work with categorical features directly instead of dummies, you will have to either use the slow KernelExplainer or prefer other more approximative techniques such as LIME for local explanations or permutation importance for global ones




""""
##############################################
####################### UNIT TEST
##############################################
""""

#FIRST CLASS...

if not isinstance(keyword, str): raise ValueError('search keyword is not a string')



""""
##############################################
####################### SERIALISE OBJECT
##############################################
""""


#############Pickle

import pickle

with open("data/rf_model.pickle", "wb") as f:
    pickle.dump(rfc, f)

with open("data/rf_model.pickle", "rb") as f:
    rf_pickle = pickle.load(f)
    

#You can directly use the model because it has already been trained
rf_pickle.predict(X_test)


#############Joblib

import joblib

joblib.dump(rfc, 'data/rf_model.joblib')

rf_joblib = joblib.load('data/rf_model.joblib')

#You can directly use the model because it has already been trained
rf_joblib.predict(X_test)



#######################################################              
#######################################################              
#######################################################              
########### Other varisou cheat sheets I find around online: 
#######################################################              
#######################################################              
#######################################################              


              
              
#######################################################              
######### Grid Search cheat sheet
#######################################################              


https://medium.com/swlh/the-hyperparameter-cheat-sheet-770f1fed32ff


#######################################################              
######### DS statistics mathematics cheat sheet
#######################################################              

https://medium.com/analytics-vidhya/your-ultimate-data-science-statistics-mathematics-cheat-sheet-d688a48ad3db





