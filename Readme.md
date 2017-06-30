Hi everyone, 
This is Rushi Pandya and my project here is Sentiment Analysis on twitter Data.
There are two parts of the project:
1. Data collection, Cleaning and Data Classification

In this part I have pre collected data from different online sources and using various classifiers from the SCIKIT learn 
I have classified the data into positive, negative and neutral. This classification works on the ratio of 4:5 which means 
4 parts of raw data is used for the training the classifier and the last part of the data used for the testing the classifier.
I have the accuracy of this classification upto 78%.

Tools I have used for this classification are:
python programming language and the different packages from python programming language they are shown below:
1. pip
2. sklearn
3. numpy
4. matplotlib
5. mongodb (which will be explained in the later part)
6. basemap
7. geopy
8. pymong
9. nltk
10. twitter
11. tweepy
12. pandas

2. Data Analysis over the live tweets using the trained classifiers:

In this part I have collected the live twitter data over a particular keyword which can be very wide range starting from sports,
current affairs of a government, TV shows or it can be anything. I have only collected tweets based on English language to make 
the analysis understadable for the users. Now, I have stored the tweets into mongodb after cleaning them and then analyzed these
with he classifier I trained and tested. I have also collected the locations form the tweets and I have made a list of some of the 
pre-defined places from Unnited states and compared the locations with the list and plotted the location based data for example
green dot for positive, yellow dot for neutral and a red dot for negative.

I have also collected the score from the tweets based on a overall polarity of the tweet in which I have compared the polar words
to list of 12K words I got from Stanford University with their polarities. I have produced the live graph from this polarities.

There are some of the difficulties I have faced during the project were if you're using MACOSX then there's possibility that 
matplotlib module might not work properly with the MACOSX and my geotagging project is still in beta version and it might not
give accurate results.


