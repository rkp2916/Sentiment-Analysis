import csv
import re
import sys
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

""" Analyzes the large feed of data and returns a sentiment value for each tweet."""
reload(sys)
sys.setdefaultencoding('utf-8')

def get_stop_word_list(stopWordListFileName):
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')
    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

stopWords = get_stop_word_list('data/stopwords.txt')

#Already in my pre_process.py so unnecessary here
def process_tweet(tweet):
    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # Trim quotes
    tweet = tweet.strip('\'"')
    return tweet


def get_features_from_tweet(tweet):
    tokens = process_tweet(tweet).split()
    return dict((w, True) for w in tokens)


def get_train_features_from_tweets(tweets, pos_neg):
    tweet_features = []
    for tweet in tweets:
        features = get_features_from_tweet(tweet)
        tweet_features.append((features, pos_neg))
    return tweet_features


pos_tweets = []
neg_tweets = []
neu_tweets = []


raw_tweets = csv.reader(open('data/training_neatfile.csv', 'rb'), delimiter=',')
tweets = []
for row in raw_tweets:
    sentiment = row[0]
    tweet = row[1]
    # processedTweet = process_tweet(tweet)
    # featureVector = getFeatureVector(processedTweet, stopWords)
    item = (tweet, sentiment)
    tweets.append(item)
    if sentiment == "positive":
        pos_tweets += item
    elif sentiment == "neutral":
        neu_tweets += item
    else:
        neg_tweets += item

negcutoff, poscutoff, neucutoff = len(neg_tweets) * 4 / 5, len(pos_tweets) * 4 / 5, len(neu_tweets) * 4 / 5
pos_train, pos_test = pos_tweets[:poscutoff], pos_tweets[poscutoff:]
neg_train, neg_test = neg_tweets[:negcutoff], neg_tweets[negcutoff:]
neu_train, neu_test = neu_tweets[:neucutoff], neu_tweets[neucutoff:]
neg_feats_train = get_train_features_from_tweets(neg_train, 'negative')
pos_feats_train = get_train_features_from_tweets(pos_train, 'positive')
neu_feats_train = get_train_features_from_tweets(neg_train, 'neutral')

train_feats = neg_feats_train + pos_feats_train + neu_feats_train

naive_bayes_classifier = SklearnClassifier(BernoulliNB())
naive_bayes_classifier.train(train_feats)

# Evaluation
correct, wrong = 0, 0

for tweet in neg_test:
    features = get_features_from_tweet(tweet)
    result = naive_bayes_classifier.classify(features)
    if result == "negative":
        correct += 1
    else:
        wrong += 1


for tweet in pos_test:
    features = get_features_from_tweet(tweet)
    result = naive_bayes_classifier.classify(features)
    if result == "positive":
        correct += 1
    else:
        wrong += 1

for tweet in neu_test:
    features = get_features_from_tweet(tweet)
    result = naive_bayes_classifier.classify(features)
    if result == "neutral":
        correct += 1
    else:
        wrong += 1

#print "Accuracy: {}".format(correct / float(correct + wrong))

"""The trained classifier is then tested on the newly acquired live feed data set"""

testTweet = open('file.csv', 'rb')
for item in testTweet:
    processedTestTweet = process_tweet(item)
    print item, naive_bayes_classifier.classify(get_features_from_tweet(processedTestTweet))
    #print "Accuracy: {}".format(correct / float(correct + wrong))