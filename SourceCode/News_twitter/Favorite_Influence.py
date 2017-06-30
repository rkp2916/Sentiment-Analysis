from pymongo import MongoClient
from pylab import *

client=MongoClient("localhost",27017)
db = client.twitter
#news = db.news

records = db.news.aggregate([
{"$match":{"$and": [{"tokens":"syria"}, {"wl_score": 0}]}},
{"$group":{"_id":"$author_name","total":{"$sum":"$status_favorite_count"}}}
])
print records
list_total = []
list_author = []
results = []
results = list(records)
for i in results:
	list_total.append(float(i['total']))
	list_author.append(i['_id'])
print;print
print zip(list_author,list_total)

s = sum(list_total)
print s
list_total = [float(i*100/s) for i in list_total]
print list_total

figure(1, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])

pie(list_total, labels=list_author,
                autopct='%1.4f%%', shadow=True)

title('Influential chart on the basis of favorites', bbox={'facecolor':'0.8', 'pad':5})

show()
