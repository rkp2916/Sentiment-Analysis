# -*- coding: utf-8 -*-

import sys
import json
import twitter
import tweepy
import datetime
import re
import urllib2
from BeautifulSoup import BeautifulSoup as bs
from recipe__oauth_login import oauth_login
from pymongo import *
from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords
import string

#TWEET_ID = 57305746 # Example: 24877908333961216

#myfile=open("RKP1107_following_ids1.txt","r")
#idlist=myfile.readlines()
client = MongoClient("localhost", 27017)
db = client.twitter

# word list
filenameWords = 'words/Words.txt'
wordlist = dict(map(lambda y: (y[0], float(y[1])), [
            y.strip().split() for y in open(filenameWords) ]))

# Word splitter pattern
pattern_split = re.compile(r"\W+")

SCREEN_NAME="nytimes"
consumer_key="gNZackrGSYgUk4uOoWhUyEerp"
consumer_secret='ITu6JYRmNMWxSjjXZYk3x1Ww0oi16Qhrsdm8a5W8AV3kR5r5cT'
access_key='474953969-ZOnbTKwy2inb2kZOIbgGbwr0C6Q8QTaFsDe8NrBG'
access_secret='IBcnBjnfk00FUrYHMxmEyJfHQfkaEtKpd0dEdSqNVUsHp'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

list1 = ['author', 'contributors', 'coordinates', 'created_at', 'destroy', 'entities', 'favorite', 'favorite_count', 'favorited', 'filter_level', 'geo', 'id', 'id_str', 'in_reply_to_screen_name', 'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'lang', 'parse', 'parse_list', 'place', 'possibly_sensitive', 'retweet', 'retweet_count', 'retweeted', 'retweets', 'source', 'source_url', 'text', 'truncated', 'user']

#x = 0
#while 1:
try:	
	tweet = api._lookup_users(screen_name = SCREEN_NAME)
	for t in tweet:
		id1=t.id
#	if not idlist: break	
	tweet = api._lookup_users(user_id = id1)
	for t in tweet:
		screen_name=t.screen_name
		print screen_name
		
		page_list = []
		n = 0
		myfile = open(screen_name,"wb")
		try:
			result = tweepy.Cursor(api.user_timeline, count=200,user_id = id1, include_rts = True).pages(2)
		except:
			x = True
			while x:
				try:
					print 'Waiting for internet connection...'
					x = False
					urllib2.urlopen("http://google.com", timeout = 1)
				except:
					x = True
			continue
		for page in result:
			page_list.append(page)
			n = n+1
			print n
			for page in page_list:
				for status in page:
#					print dir(status)
					author_description = status.author.description
					author_id_str = status.author.id_str
					author_favourites_count = status.author.favourites_count
					author_name = status.author.name
					author_screen_name = status.author.screen_name
					author_followers_count = status.author.followers_count
					author_location = status.author.location
					author_friends_count = status.author.friends_count
					author_url = status.author.url
					author_verified = status.author.verified
					author_statuses_count = status.author.statuses_count
					status_id_str = status.id_str
					d = status.created_at
					status_created_at_date_str = status.created_at.strftime("%d-%m-%Y")
					status_created_at_time_str = status.created_at.strftime("%H:%M:%S")
					status_entities = status.entities
					status_favorite_count = status.favorite_count
					status_in_reply_to_screen_name = status.in_reply_to_screen_name
					status_in_reply_to_status_id_str = status.in_reply_to_status_id_str
					status_in_reply_to_user_id_str = status.in_reply_to_user_id_str
					status_retweet_count = status.retweet_count
					status_source = status.source
					status_text = str(status.text.encode('ascii','ignore'))

					str1 = status_text.translate(string.maketrans("",""), string.punctuation)
#					print 'Removing punctuation marks'

					str1 = str1.lower()
#					print 'Normalizing to lower case:'
	
					lst = nltk.word_tokenize(str1)
#					print 'After tokenization:'
				
					minlength = 2
					lst = [token for token in lst if(not token in stopwords.words('english')) and len(token) >= minlength]
#					print 'Removing Stop Words:'
		
					sentiments = map(lambda word: wordlist.get(word, 0), lst)
#					print zip(lst,sentiments)
					if sentiments:
						# How should you weight the individual word sentiments? 
						# You could do N, sqrt(N) or 1 for example. Here I use N
						sentiment = float(sum(sentiments))/len(lst)
					else:
					        sentiment = 0
					print status_text
					print 'Tweet: '+ str(sentiment)

					news_content = ''					
					news_title = ''
					lst1 = []
					lst2 = []
					try:
						if status.entities['urls']:
							ex_url = status.entities['urls'][0]['expanded_url']
							html = urllib2.urlopen(ex_url)
							soup = bs(html)
							tds1 = soup.find('h1')
							news_title = tds1.text
							tds = soup.findAll('div',attrs={"class" : "Normal"})
							for td in tds:
								news_content += ' '+td.text
							news_content = str(news_content.encode('ascii','ignore'))
#							print news_content
							str2 = news_content.translate(string.maketrans("",""), string.punctuation)
	
							str1 = news_title.lower()
							lst1 = nltk.word_tokenize(str1)
#							print 'Removing punctuation marks'
	
							str2 = str2.lower()
#							print 'Normalizing to lower case:'
		
							lst2 = nltk.word_tokenize(str2)
#							print 'After tokenization:'
				
							minlength = 2
							lst2 = [token for token in lst2 if(not token in stopwords.words('english')) and len(token) >= minlength]
							lst1 = [token for token in lst1 if(not token in stopwords.words('english')) and len(token) >= minlength]
#							print lst1
#							print lst2
#							print 'Removing Stop Words:'

							sentiments = map(lambda word: wordlist.get(word, 0), lst2)
				
							if sentiments:
							# How should you weight the individual word sentiments? 
							# You could do N, sqrt(N) or 1 for example. Here I use sqrt(N)
								sentiment = sum(sentiments)/math.sqrt(len(sentiments))
							else:
							        sentiment = 0
							print 'Article: ' + str(sentiment)
					except:
						continue
					db.news.insert({'author_description':author_description,'author_id_str':author_id_str,'author_favourites_count':author_favourites_count,'author_name':author_name,'author_screen_name':author_screen_name,'author_followers_count':author_followers_count,'author_location':author_location,'author_friends_count':author_friends_count,'author_url':author_url,'author_verified':author_verified,'author_statuses_count':author_statuses_count,'status_id_str':status_id_str,'status_created_at':d,'status_created_at_date_str':status_created_at_date_str,'status_created_at_time_str':status_created_at_time_str,'status_entities':status_entities,'status_favorite_count':status_favorite_count,'status_in_reply_to_screen_name':status_in_reply_to_screen_name,'status_in_reply_to_status_id_str':status_in_reply_to_status_id_str,'status_in_reply_to_user_id_str':status_in_reply_to_user_id_str,'status_retweet_count':status_retweet_count,'status_source':status_source,'status_text':status_text,'tokens':lst,'wl_score':sentiment,'news_title':news_title,'news_content':news_content,'news_title_tokens':lst1,'news_content_tokens':lst2})
					myfile.write(status_text+"\n")
	myfile.close()
#	x = x + 1
except EOFError:
	print "End Of File"
# No authentication required, but rate limiting is enforced

#twee = t.statuses.show(id=TWEET_ID, include_entities=1) 

#s = tweet.text
#print s.encode("utf8")

#print json.dumps(twee, indent=1)
