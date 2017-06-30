import re
import logging
import time
import warnings
from collections import Counter
import numpy as np
import pandas as pd      
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB

warnings.filterwarnings('ignore')
tweet_data=pd.read_csv("train_final.csv")
 #checking how data looks like
counts = tweet_data['Sentiment'].value_counts().to_dict() 
print (counts) #let's see how many pos and neg value we have
tweet_data.head()
stemmer = SnowballStemmer("english") #stemming process
operators = set(('and', 'or', 'not'))
operators1=set(('quot','amp' )) 
stops = set(stopwords.words("english"))-operators #remove some words from stop words so we can get all features
stops.update(operators1)

print(tweet_data["SentimentText"][10])

#function that clean all tweets

def tweet_clean(tweets):
    user_clean = re.sub('@[^\s]+', ' ', tweets)
    link_clean  = re.sub('((www\.[\s]+)|(https?://[^\s]+))', '', user_clean)
    letters_only = re.sub("[^a-zA-Z]", " ", link_clean) 
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stops]
    final_words=[stemmer.stem(word) for word in meaningful_words]
    return (" ".join(final_words))

num_tweeet = tweet_data["SentimentText"].size
print("Cleaning and parsing the training set tweets...\n")
tweet_clean_feature = []
for i in range( 0, num_tweeet ):
    if( (i+1)%10000 == 0 ):
        print("Review %d of %d" % ( i+1, num_tweeet ))                                                                 
    tweet_clean_feature.append( tweet_clean( tweet_data["SentimentText"][i] ))
tweet_data["CleanText"]=tweet_clean_feature
tweet_data.head(10)

tweet_sample=tweet_data.sample(frac=0.27)
tweet_sample.shape
tweet_sample = tweet_sample.reset_index(drop=True)
X=tweet_sample.CleanText
y=tweet_sample.Sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3,test_size=0.20)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
tweet_sample = tweet_sample.reset_index(drop=True)
X=tweet_sample.CleanText
y=tweet_sample.Sentiment
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer = CountVectorizer(analyzer = "word",ngram_range=(1,2)) #bi-gram
vectorizer1 = CountVectorizer(analyzer = "word",ngram_range=(1,1)) #uni-gram
tf_transformer = TfidfTransformer(use_idf=False)
text_clf = Pipeline([('vect', vectorizer),
                     ('tfidf', tf_transformer),
                     ('clf', RandomForestClassifier(n_estimators =120, n_jobs=-1,verbose=3))])
text_clf.fit(X_train,y_train)
predicted = text_clf.predict(X_test)


score1=np.mean(predicted == y_test)
conf=metrics.confusion_matrix(y_test, predicted)
df_cm = pd.DataFrame(conf, index = [i for i in ["Actual Neg","Actual Pos"]],
                     columns = [i for i in ["Predict Neg","Predict Pos"]])

sn.plt.figure(figsize = (10,7))
sn.plt.title("Confuse Matrix for Random Forest with Bag of bi-gram Model",fontsize=16)
sn.heatmap(df_cm, annot=True,fmt='g',annot_kws={"size": 16})
plt.show()
text_clf2 = Pipeline([('vect', vectorizer1),
                     ('tfidf', tf_transformer),
                     ('clf', RandomForestClassifier(n_estimators =120, n_jobs=-1,verbose=3))])
text_clf2.fit(X_train,y_train)
predicted1 = text_clf2.predict(X_test)
score2=np.mean(predicted1 == y_test)
conf1=metrics.confusion_matrix(y_test, predicted1)
df_cm1 = pd.DataFrame(conf1, index = [i for i in ["Actual Neg","Actual Pos"]],
                     columns = [i for i in ["Predict Neg","Predict Pos"]])

sn.plt.figure(figsize = (10,7))
sn.plt.title("Confuse Matrix for Random Forest with Bag of uni-gram Model",fontsize=16)
sn.heatmap(df_cm1, annot=True,fmt='g',annot_kws={"size": 16})
plt.show()
text_clf3 = Pipeline([('vect', vectorizer),
                     ('tfidf', tf_transformer),
                     ('clf', RandomForestClassifier(n_estimators =200, n_jobs=-1,verbose=3,max_features="sqrt"))])
text_clf3.fit(X_train,y_train)
text_clf4 = Pipeline([('vect', vectorizer1),
                     ('tfidf', tf_transformer),
                     ('clf', RandomForestClassifier(n_estimators =200, n_jobs=-1,verbose=3,max_features="sqrt"))])
text_clf4.fit(X_train,y_train)
predicted2 = text_clf3.predict(X_test)
score3=np.mean(predicted2 == y_test)
conf2=metrics.confusion_matrix(y_test, predicted2)
df_cm2 = pd.DataFrame(conf2, index = [i for i in ["Actual Neg","Actual Pos"]],
                     columns = [i for i in ["Predict Neg","Predict Pos"]])

sn.plt.figure(figsize = (10,7))
sn.plt.title("Confuse Matrix for Random Forest with Bag of bi-gram Model with sqrt features",fontsize=16)
sn.heatmap(df_cm2, annot=True,fmt='g',annot_kws={"size": 16})
plt.show()
predicted3 = text_clf4.predict(X_test)
score4=np.mean(predicted3 == y_test)
conf3=metrics.confusion_matrix(y_test, predicted3)
df_cm3 = pd.DataFrame(conf3, index = [i for i in ["Actual Neg","Actual Pos"]],
                     columns = [i for i in ["Predict Neg","Predict Pos"]])

sn.plt.figure(figsize = (10,7))
sn.plt.title("Confuse Matrix for Random Forest with Bag of uni-gram Model with sqrt features",fontsize=16)
sn.heatmap(df_cm3, annot=True,fmt='g',annot_kws={"size": 16})
plt.show()
a=[score1,score2,score3,score4]
y= np.array(a)*100
x=["Bi-gram","Uni-gram","SqrtBi","SqrtUni"]
sns.set(style="darkgrid", context="talk")
plt.ylim(70, 78)
colors =['grey' if (x < max(y)) else 'red' for x in y ]
sn.plt.title("Random Forest Classifier",fontsize=16)
g = sn.barplot(x,y,palette=colors)
g.set(xlabel='Diffrent Parameter', ylabel='Accurancy')
plt.show()
smit=vectorizer.fit_transform(X_train)
X_train_tf=tf_transformer.fit_transform(smit)

from sklearn.svm import SVC

tuned_parameters = [{'C': [1, 10, 100, 1000]}]
clf=svm.LinearSVC()

CV_rfc1 = GridSearchCV(estimator=clf, param_grid=tuned_parameters, cv= 5)
CV_rfc1.fit(X_train_tf, y_train)
print(CV_rfc1.best_params_)
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
text_clf_ = Pipeline([('vect', vectorizer),
                     ('tfidf', tf_transformer),
                      ('clf',LinearSVC(C=1))])
                     
text_clf_.fit(X_train,y_train)
predicted_ = text_clf_.predict(X_test)
score5=np.mean(predicted_ == y_test)
conf4=metrics.confusion_matrix(y_test, predicted_)
df_cm4 = pd.DataFrame(conf4, index = [i for i in ["Actual Neg","Actual Pos"]],
                     columns = [i for i in ["Predict Neg","Predict Pos"]])

sn.plt.figure(figsize = (10,7))
sn.plt.title("Confuse Matrix for SVM(uni-gram Model) with C=1",fontsize=16)
sn.heatmap(df_cm4, annot=True,fmt='g',annot_kws={"size": 16})
plt.show()
text_clf_2 = Pipeline([('vect', vectorizer),
                     ('tfidf', tf_transformer),
                      ('clf',LinearSVC(C=1000))])
                     
text_clf_2.fit(X_train,y_train)
predicted_1 = text_clf_2.predict(X_test)
score6=np.mean(predicted_1 == y_test)
conf5=metrics.confusion_matrix(y_test, predicted_1)
df_cm5 = pd.DataFrame(conf5, index = [i for i in ["Actual Neg","Actual Pos"]],
                     columns = [i for i in ["Predict Neg","Predict Pos"]])

sn.plt.figure(figsize = (10,7))
sn.plt.title("Confuse Matrix for SVM(uni-gram Model) with C=1000",fontsize=16)
sn.heatmap(df_cm5, annot=True,fmt='g',annot_kws={"size": 16})
plt.show()
text_clf_3 = Pipeline([('vect', vectorizer),
                     ('tfidf', tf_transformer),
                      ('clf',LinearSVC(C=100))])
                     
text_clf_3.fit(X_train,y_train)
predicted_2 = text_clf_3.predict(X_test)
score7=np.mean(predicted_2 == y_test)
conf6=metrics.confusion_matrix(y_test, predicted_2)
df_cm6 = pd.DataFrame(conf6, index = [i for i in ["Actual Neg","Actual Pos"]],
                     columns = [i for i in ["Predict Neg","Predict Pos"]])

sn.plt.figure(figsize = (10,7))
sn.plt.title("Confuse Matrix for SVM(uni-gram Model) with C=100",fontsize=16)
sn.heatmap(df_cm6, annot=True,fmt='g',annot_kws={"size": 16})
plt.show()
score7=np.mean(predicted_2 == y_test)
conf6=metrics.confusion_matrix(y_test, predicted_2)
df_cm6 = pd.DataFrame(conf6, index = [i for i in ["Actual Neg","Actual Pos"]],
                     columns = [i for i in ["Predict Neg","Predict Pos"]])

sn.plt.figure(figsize = (10,7))
sn.plt.title("Confuse Matrix for SVM(uni-gram Model) with C=100",fontsize=16)
sn.heatmap(df_cm6, annot=True,fmt='g',annot_kws={"size": 16})
plt.show()
predicted_3 = text_clf_4.predict(X_test)
score8=np.mean(predicted_3 == y_test)
conf7=metrics.confusion_matrix(y_test, predicted_3)
df_cm7 = pd.DataFrame(conf7, index = [i for i in ["Actual Neg","Actual Pos"]],
                     columns = [i for i in ["Predict Neg","Predict Pos"]])

sn.plt.figure(figsize = (10,7))
sn.plt.title("Confuse Matrix for SVM(bi-gram Model) with C=1",fontsize=16)
sn.heatmap(df_cm7, annot=True,fmt='g',annot_kws={"size": 16})
plt.show()
a=[score5,score6,score7,score8]
b= np.array(a)*100
c=["C=1","C=100","C=1000","UniGram"]
sns.set(style="darkgrid", context="talk")
plt.ylim(70, 78)
colors =['grey' if (x < max(b)) else 'red' for x in b ]
sn.plt.title("Linear SVM",fontsize=16)
g = sn.barplot(c,b,palette=colors)
g.set(xlabel='Diffrent Parameter', ylabel='Accurancy')
plt.show()
X_testing=vectorizer.fit_transform(X)
X_test_tf=tf_transformer.fit_transform(X_testing)
clf=LinearSVC(C=1)

scores1 = cross_val_score(clf,X_test_tf, y, cv=10, scoring='accuracy')
x=[1,2,3,4,5,6,7,8,9,10]
sns.set(style="darkgrid", context="talk")
plt.ylim(70, 78)
colors =['grey' if (x < max(scores1)) else 'red' for x in scores1 ]
# sn.plt.title("Linear SVM",fontsize=16)
sn.barplot(x,scores1*100,palette=colors)
X_testing1=vectorizer1.fit_transform(X)
X_test_tf1=tf_transformer.fit_transform(X_testing1)
# clf=LinearSVC(C=1)

scores2 = cross_val_score(clf,X_test_tf1, y, cv=10, scoring='accuracy')
sns.set(style="darkgrid", context="talk")
plt.ylim(70, 78)
colors =['grey' if (x < max(scores2)) else 'red' for x in scores2 ]
# sn.plt.title("Linear SVM",fontsize=16)
sn.barplot(x,scores2*100,palette=colors)
Final_y=[score1,score2,score3,score4,score5,score6,score7,score8]
Final_y=np.array(Final_y)*100
Final_x=["Bi-gram","Uni-gram","SqrtBi","SqrtUni","C=1","C=100","C=1000","UniGramSVM"]
sns.set(style="darkgrid", context="talk")
plt.ylim(72, 78)
colors =['grey' if (x < max(Final_y)) else 'red' for x in Final_y ]
sn.plt.title("RandomForest + SVM",fontsize=16)
g = sn.barplot(Final_x,Final_y,palette=colors)
g.set(xlabel='Diffrent Parameter', ylabel='Accurancy')
plt.show()    


text_clf_nb = Pipeline([('vect', vectorizer1),
                     ('tfidf', tf_transformer),
                      ('clf',MultinomialNB(fit_prior=True))])
                     
text_clf_nb.fit(X_train,y_train)    
result=text_clf_nb.predict(X_test)
np.mean(result == y_test)
result=text_clf_nb.predict(X_test)
np.mean(result == y_test)
text_clf_nb1 = Pipeline([('vect', vectorizer),
                         ('tfidf', tf_transformer),
                         ('clf',BernoulliNB())])
                     
text_clf_nb1.fit(X_train,y_train)
result2=text_clf_nb1.predict(X_test)
score9=np.mean(result2==y_test)

text_clf_nb2 = Pipeline([('vect', vectorizer),
                         ('tfidf', tf_transformer),
                         ('clf',MultinomialNB())])
                     
text_clf_nb2.fit(X_train,y_train)
result3=text_clf_nb2.predict(X_test)
score10=np.mean(result3==y_test)

text_clf_nb3 = Pipeline([('vect', vectorizer1),
                         ('tfidf', tf_transformer),
                         ('clf',MultinomialNB())])
                     
text_clf_nb3.fit(X_train,y_train)
result4=text_clf_nb3.predict(X_test)
score11=np.mean(result4==y_test)
text_clf_nb4 = Pipeline([('vect', vectorizer1),
                         ('tfidf', tf_transformer),
                         ('clf',BernoulliNB())])
                     
text_clf_nb4.fit(X_train,y_train)
result5=text_clf_nb4.predict(X_test)
score12=np.mean(result5==y_test)
Final_y=[score9,score10,score11,score12]
Final_y=np.array(Final_y)*100
Final_x=["BernoulliNB","MultinomialNB","BernoulliNB-Uni","MultinomialNB-Uni"]
sn.set(style="darkgrid", context="talk")
plt.ylim(72, 78)
colors =['grey' if (x < max(Final_y)) else 'red' for x in Final_y ]
sn.plt.title("Naive Bayes",fontsize=16)
g = sn.barplot(Final_x,Final_y,palette=colors)
g.set(xlabel='Diffrent Parameter', ylabel='Accurancy')
plt.show() 