import re
import geopy
import better_exceptions
from geopy import geocoders
from geopy.geocoders import Nominatim
import json
import numpy as np
import pandas as pd
import pymongo
from pymongo import MongoClient
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits import basemap
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

client =  MongoClient('localhost', 27017)
db = client.twitter

#remove existing data
db.latlong.remove()

cursor = db.streaming.find({"$or" :[{"coordinates": "New York, NY"}, {"coordinates" : "Washington, DC"}, {"coordinates" : "Chicago, IL"}, {"coordinates": "San Francisco, CA"}, {"coordinates": "Houston, TX"},
                             {"coordinates": "Dallas, TX"}, {"coordinates": "Nashville, TN"}, {"coordinates": "Columbus, OH"}, {"coordinates": "Cleveland, OH"}, {"coordinates": "Seattle, WA"}, {"coordinates": "Albany, NY"},
                             {"coordinates": "San Antonio, TX"},{"coordinates": "San Diego, CA"}, {"coordinates": "Austin, TX"}, {"coordinates": "Denver, CO"},{"coordinates": "Los Angeles, CA"}, {"coordinates": "Miami, FL"}, 
                             {"coordinates": "Portland, OR"}, {"coordinates": "Philadelphia, PA"}, {"coordinates": "New Orleans, LA "}, {"coordinates": "Boston, MA"}, {"coordinates": "Atlanta, GA"}, {"coordinates": "Phoenix, AZ"},
                             {"coordinates": "Pittsburgh, PA"}, {"coordinates": "San Jose, CA"}, {"coordinates": "Kansas City, MO"}, {"coordinates": "Baltimore, MD"}, {"coordinates": "Indianapolis, IN"}, {"coordinates": "Tempa, FL"},
                             {"coordinates": "Las Vegas, NV"}, {"coordinates": "Milwaukee, WI"}, {"coordinates": "Madison, WI"}, {"coordinates": "Charlotte, NC"}, {"coordinates": "Boise, ID"}, {"coordinates": "Charleston, SC"},
                             {"coordinates": "Cincinati, OH"}, {"coordinates": "Scottsdale, AZ"}, {"coordinates": "Oklahoma City, OK"}, {"coordinates": "Portland, OR"}, {"coordinates": "Omaha, NE"}, {"coodinates": "Albuquerque, NM"},
                             {"coordinates": "Detroit, MI"}, {"coordinates": "Minneapolis, MN"}, {"coordinates": "Saint Paul, MN"}, {"coordinates": "St. Cloud, MN"}, {"coordinates": "Portland, ME"},{"coordinates": "Sioux Falls, SD"},
                             {"coordinates": "Richmond, VA"}, {"coordinates": "Charleston, WV"}, {"coordinates": "St. Louis, MO"}, {"coordinates": "Wichita, KS"}, {"coordinates": "Bismarck, ND"}, {"coordinates": "Fargo, ND"},
                             {"coordinates": "Manchester, NH"},{"coordinates": "Concord, NH"}, {"coordinates": "Burlington, VT"}, {"coordinates": "Tempa, FL"},{"coordinates": "Wichita, KS"}, {"coordinates": "Trenton, NJ"},{"coordinates": "Birmingham, AL"},
                             {"coordinates": "Montgomery, AL"}, {"coordinates": "Wichita, KS"}, {"coordinates": "Billings, MT"}, {"coordinates": "Helena, MT"}, {"coordinates": "Cheyenne, WY"}, {"coordinates": "Salt Lake City, UT"},
                             {"coordinates": "Des Moines, IA"}, {"coordinates": "Little Rock, AR"}, {"coordinates": "Green Bay, WI"}, {"coordinates": "Jackson, MS"}, {"coordinates": "Louisville, KY "}, {"coordinates": "Frankfort, KY"},
                             {"coordinates": "Wilmington, DE"}, {"coordinates": "Dover, DE"}, {"coordinates": "Providence, RI"}, {"coordinates": "Newport, RI"}, {"coordinates": "Warwick, RI"}, {"coordinates": "Memphis, TN"}, {"coordinates": "Newark, NJ"},
                             {"coordinates": "Jersey City, NJ"}, {"coordinates": "Paterson, NJ"}, {"coordinates": "Buffalo, NY"}, {"coordinates": "Rochester, NY"}, {"coordinates": "Syracuse, NY"}, {"coordinates": "Springfield, IL"}, {"coordinates": "Naperville, IL"},
                             {"coordinates": "Worcester, MA"}, {"coordinates": "Juneau, AK"}, {"coordinates": "Anchorage, AK"}, {"coordinates": "Honolulu, HI"}]})
        
tweet_fields = ['coordinates']
df = pd.DataFrame(list(cursor), columns = tweet_fields)
for each in df:
    print(df)
    
gn = geocoders.GeoNames(username= "rushi29")
gn.geocode(df.ix[0],exactly_one=False)
    

locations = dict()
for i in range(len(df)):
    loc = df.ix[i].coordinates
    #db.loc.insert({"location" : loc})
    if loc is None:continue
    try:
        print(i,type(loc),loc)
        loc = loc.decode()
        tmp = gn.geocode(loc)
    except:continue
    if tmp is not None:
        locations[loc] = (tmp.latitude,tmp.longitude)
        db.latlong.insert({'latitude': tmp.latitude, 'longitude': tmp.longitude, 'location': loc})
print(locations,len(locations))



