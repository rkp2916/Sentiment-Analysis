import sys
import tweepy
import pymongo
from pymongo import MongoClient
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import math
import csv
import re



with open("BBCNews", "r") as f:
	text = f.read()
	text = re.sub(r"http\S+", "", text)
	text = re.compile('RT @').sub('',text).strip()
	text = re.compile('via').sub('', text).strip()
	text = re.compile('say').sub('',text).strip()
	#text = lambda x: re.compile('\#').sub('', re.compile('RT @').sub('@', x, count=1).strip())
	text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
	text = ' '.join(re.sub("(BBC[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
wordcloud = WordCloud(font_path='/Library/Fonts/Arial Black.ttf',
                          stopwords=STOPWORDS,
                          background_color='white',
                          width=1200,
                          height=1000
                         ).generate(text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()



	





