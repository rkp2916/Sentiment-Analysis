import math
import re
import os
import sys
import pymongo
from pymongo import MongoClient
import pandas as pd

reload(sys)
sys.setdefaultencoding('utf-8')

#Open a connection to MongoDb (localhost)
client=MongoClient("localhost", 27017)
db=client.twitter

filenameWords = 'words/wordweightings.txt'
wordlist = dict(map(lambda (w,s) : (w, float(s)), [
    ws.strip().split('\t') for ws in open(filenameWords) ]))
# Word splitter pattern
pattern_split = re.compile(r"\W+")

def sentiment(text):
    """
    Returns a float for sentiment strength based on the input text.
    Positive values are positive valence, negative value are negative valence.
    """
    words = pattern_split.split(text.lower())
    sentiments1 = map(lambda word: wordlist.get(word, 0), words)
    sentiments = []
    for k in sentiments1:
	if k != 0:
		sentiments.append(k)
    if sentiments:
        # How should you weight the individual word sentiments?
        # You could do N, sqrt(N) or 1 for example. Here I use sqrt(N
        sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))
    else:
        sentiment = 0
    #print sentiment
    return sentiment

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

sentiments = map(sentiment, [ tweet['text'] for tweet in cursor ])
j = sum(sentiments)/math.sqrt(len(sentiments))
