import sys
import tweepy
import codecs
#from recipe__oauth
from recipe__search import *
import csv
from pymongo import MongoClient
from pymongo import *
#'''


reload(sys)
sys.setdefaultencoding('utf-8')
#Open a connection to MongoDb (localhost)
client =  MongoClient("localhost", 27017)
db = client.twitter

#Remove any existing data
db.streaming.remove()

# Query terms
#loc = -38, 16, -22, 37  #-122.75,36.8,-121.75,37.8,-74,40,-73,41
#Q = sys.argv[1:] 

# Get these values from your application settings
CONSUMER_KEY = 'gNZackrGSYgUk4uOoWhUyEerp'
CONSUMER_SECRET = 'ITu6JYRmNMWxSjjXZYk3x1Ww0oi16Qhrsdm8a5W8AV3kR5r5cT'
# Get these values from the "My Access Token" link located in the
# margin of your application details, or perform the full OAuth
# dance

ACCESS_TOKEN='474953969-ZOnbTKwy2inb2kZOIbgGbwr0C6Q8QTaFsDe8NrBG'
ACCESS_TOKEN_SECRET='IBcnBjnfk00FUrYHMxmEyJfHQfkaEtKpd0dEdSqNVUsHp'
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)


# Note: Had you wanted to perform the full OAuth dance instead of using
# an access key and access secret, you could have uses the following 
# four lines of code instead of the previous line that manually set the
# access token via auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
# 
#auth_url = auth.get_authorization_url(signin_with_twitter=True)
# webbrowser.open(auth_url)on
# verifier = raw_input('PIN: ').strip()
# auth.get_access_token(verifier)

handle=csv.writer(open('file1.csv','wb'))



class CustomStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        try:
            db.streaming.insert({'text': status.text, 'coordinates':status.author.location})
            handle.writerow([status.text])
            print status.text
            print status.author.location
        except Exception, e:
            print >> sys.stderr, 'Encountered Exception:', e
            pass

    def on_error(self, status_code):
        print status_code
        #print >> sys.stderr, 'Encountered error with status code:', status_code
        return True # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True # Don't kill the stream   

# Create a streaming API and set a timeout value of 60 seconds
# Use OAuth to authenticate.
streaming_api = tweepy.streaming.Stream(auth, CustomStreamListener(), timeout=60)
#places = api.geo_search(query="USA", granularity="country")

# Optionally filter the statuses you want to track by providing a list
# of users to "follow"

print 'Filtering the public timeline for "%s"' % (' '.join(sys.argv[1:]),)
#setTerms = ['Modi','Narendra Modi','NaMo','NarendraModi']
#setTerms = ['Arsenal','Champions League','ChampionsLeague']
#setTerms = ['Rahul Gandhi','RahulGandhi','RahulSpeaksToArnab','Rahul Speaks To Arnab','ArnabVsRahul','Arnab Vs Rahul']
#setTerms = ['Paul Walker','PaulWalker']
#setTerms = ['Nelson Mandela','NelsonMandela']
#setTerms = ['Dortmund']
setTerms = ['RHOA']
#setTerms = ['Kejriwal']
#setTerms = ['Arvind Kejriwal','Arvind','Kejrival', 'Kejriwal']
#setTerms = ['Sunanda Pushkar', 'Shashi Tharoor', 'SunandaPushkar', 'ShashiTharoor', 'Sunanda', 'Shashi', 'Tharoor']
#setTerms = ['Team India','TeamIndia']
fil = open('RecentHarvest.txt','wb')
fil.write(setTerms[0])
fil.close()
streaming_api.filter(follow=None,track=setTerms, languages= ['en'], locations=[-125,25,-65,48], async=False)



#l = StreamListener()
#streamer = tweepy.Stream(auth=auth1, listener=l)#, timeout=3000000000 )
#setTerms = ['#tcot']   #Can include a comma seperated list of any number of topics in which you are interested
#setTerms = ['Fast & Furious 6','Fast and Furious 6','Fast and The Furious 6','fast and the furious 6','fast & the furious 6']
#streamer.filter(None,setTerms)


'''
l = StreamListener()
streamer = tweepy.Stream(auth=auth1, listener=l)#, timeout=3000000000 )
#setTerms = ['#tcot']   #Can include a comma seperated list of any number of topics in which you are interested
setTerms = ['Fast & Furious 6','Fast and Furious 6','Fast and The Furious 6','fast and the furious 6','fast & the furious 6']
streamer.filter(None,setTerms)
'''
