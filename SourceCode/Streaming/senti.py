import math
import sys
import pymongo
from pymongo import MongoClient
import pandas as pd
from SS import *

reload(sys)
sys.setdefaultencoding('utf-8')

#Open a connection to MongoDb (localhost)
client=MongoClient("localhost", 27017)
db=client.twitter

#remove existing data
db.senti.remove()

cursor = db.streaming.find({"$or" :[{"coordinates": "New York, NY"}, {"coordinates" : "Washington, DC"}, {"coordinates" : "chicago, IL"}, {"coordinates": "San Francisco, CA"}, {"coordinates": "Houston, TX"},
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
                             {"coordinates": "Wichita, KS"}, {"coordinates": "Worcester, MA"}, {"coordinates": "Juneau, AK"}, {"coordinates": "Anchorage, AK"}, {"coordinates": "Honolulu, HI"}]})

tweet_fields = ['coordinates']
df = pd.DataFrame(list(cursor), columns = tweet_fields)
#sentiments = map(sentiment, [ tweet['text'] for tweet in cursor ])
#j = sum(sentiments)/math.sqrt(len(sentiments))

loc = {}
for k in range(len(df)):
    db.senti.insert_one({'sentiments' :  sentiments[k], 'location': df.ix[k].coordinates })
    loc[k] = (df.ix[k].coordinates, sentiments[k])
print (loc)

#db.senti.insert({'Sentiments' : sentiments, 'Location': loc })
