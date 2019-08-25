from flask_pymongo import PyMongo
import pymongo
import pickle
import pandas as pd 
import numpy as np


# client = pymongo.MongoClient("mongodb+srv://admin:davian@daviandb-9rvqg.gcp.mongodb.net/test?retryWrites=true&w=majority")
client = pymongo.MongoClient('mongodb://localhost:27017/')

db = client.davian
collection_labeled = db.labeled
collection_log = db.log
# user_list = [1,2,3,6,8]
positive_label = []
negative_label = []
# for i in user_list:

positive_label += [item for item in collection_labeled.find({"$and": [{'user_id':"user"+str(3)}, {'label': 1}]})]
negative_label += [item for item in collection_labeled.find({"$and": [{'user_id':"user"+str(3)}, {"$or": [{'label':0} , {'label':-1}]}]})]

log_data = []
log_data += [item for item in collection_log.find({"$and": [{'user_id':"user"+str(3)}, {'adjective': 'ATTRACTIVE'}]})]

# for i in range(10):
# 	batch_data = [item for item in colelction_labeled.find({"$and": [{'user_id':"user"+str(3)}, {'batch': str(i)}]})]
batch_data = {}
for i in range(10):
	for item in collection_labeled.find({"$and": [{'user_id':"user"+str(3)}, {'batch': i}]}):
		batch_data[item['batch']] = int(item['time'])/1000

print(batch_data)

# print(len(positive_label))
# print(positive_label)
# print(batch_data)
