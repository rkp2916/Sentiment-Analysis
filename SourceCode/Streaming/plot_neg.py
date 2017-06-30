import re
import json
from pymongo import MongoClient
from mpl_toolkits import basemap
from mpl_toolkits.basemap import Basemap
import pandas as pd 
import numpy as np 
import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt 


client =  MongoClient('localhost', 27017)
db = client.twitter

cursor1 = db.latlong.find()
tweet_fields1 = ['latitude', 'longitude', 'location']
df1 = pd.DataFrame(list(cursor1), columns = tweet_fields1)
#print df1.head()


cursor2 = db.senti.find()
tweet_fields2 = ['sentiments','location']
df2 = pd.DataFrame(list(cursor2), columns = tweet_fields2)
#print (df2)
#figure size
fig = plt.figure(figsize=(12, 6), dpi=250)

#plotting on basemap
m = Basemap(
        projection='merc',
        llcrnrlon=-130,
        llcrnrlat=25,
        urcrnrlon=-60,
        urcrnrlat=50,
        lat_ts=0,
        resolution='l',
        suppress_ticks=True)
#lats = []
#lons = []
#x, y= m(df1['longitude'].values, df1['latitude'].values) 
#if df1['locations'].values == df2['location'].values:
x, y = m(df1['longitude'].values, df1['latitude'].values)

for j in range(len(df2)):
	if df2.ix[j].location == "Washington, DC":
		if df2.ix[j].sentiments < 0:
			m.plot(x,y, 'ro', markersize = 3)

#print(df2['sentiments'].mean())

#m.plot(x,y, 'go', markersize = 1)                       
m.drawcountries()
m.drawstates()
m.bluemarble()	

#plt.ion()

plt.show()

#if df1['locations'].values == df2['location'].values and df2['sentiments'] == 0:
#	m.plot(x,y , 'y*')



#x, y = m(df1['latitude'].values, df1['longitude'].values)