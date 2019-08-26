from flask_pymongo import PyMongo
import pymongo
import pickle
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


client_remote = pymongo.MongoClient("mongodb+srv://admin:davian@daviandb-9rvqg.gcp.mongodb.net/test?retryWrites=true&w=majority")
client_local = pymongo.MongoClient('mongodb://localhost:27017/')

db = client_remote.davian
collection_labeled = db.labeled
collection_log = db.log
user_list = [5,6]
total_data_time_per_batch = {}
total_data_count_number = {}
total_data_click_per_batch = {}

def maxNumber(data):
	temp = [item['batch'] for item in data]
	max_number = temp[len(temp)-1]
	return max_number

def countNumber(data, user):
	# print(data)
	print(len(data))
	positive_labeling = []
	negative_labeling = []
	for item in data:
		if item['label'] == 1:
			positive_labeling.append(item)
		else:
			negative_labeling.append(item)
	assert len(data) == len(positive_labeling) + len(negative_labeling)
	total_data_count_number[user] = [len(positive_labeling), len(negative_labeling), len(data)]
	print(total_data_count_number)


def clickPerBatch():

def timePerBatch(data, max_number, user):
	temp_data = {}
	for i in range(max_number+1):
		for item in collection_labeled.find({"$and": [{'user_id':user}, {'batch': i}]}):
			temp_data[item['batch']] = int(item['time'])/1000
	result = pd.DataFrame(data = {'batch' : list(temp_data.keys()), 'time': list(temp_data.values())})
	total_data_time_per_batch[user] = list(temp_data.values())
	x = result['batch']
	y = result['time']
	# plt.title(user + " time per batch")
	# plt.legend(loc='upper left', frameon=True)
	# plt.ylabel('time')
	# plt.xlabel('batch')
	# plt.plot(x,y, label=user)
	# plt.show()


total_data = []
for user in user_list:
	user = "user"+str(user)
	user_data = [item for item in collection_labeled.find({"$and": [{'user_id':user}]})]
	# max_number = maxNumber(user_data)
	# timePerBatch(user_data, max_number, user)	
	# countNumber(user_data,user)

	# total_data.extend(user_data)
#  ----------------------------------------- calculate total -------------------------------- 
# total_data_time_per_batch_df = pd.DataFrame()
# for item in total_data_time_per_batch:
# 	temp_df = pd.DataFrame(data = {item : total_data_time_per_batch[item]})
# 	total_data_time_per_batch_df = pd.concat([total_data_time_per_batch_df, temp_df], axis=1)

# total_data_count_number_df = pd.DataFrame()
# for item in total_data_count_number:
# 	temp_df = pd.DataFrame(data = {item : total_data_count_number[item]})
# 	total_data_count_number_df = pd.concat([total_data_count_number_df, temp_df], axis = 1)
# total_data_count_number_df = total_data_count_number_df.rename(index = {0: 'positive_labeling', 1: 'negative_labeling', 2: 'total_labeling'})

























# data = collection_labeled.find({"$and": [{'user_id':"user5"}]})
# temp = [item['batch'] for item in data]
# max_number = temp[len(temp)-1]
# print(max_number)

# temp_data = {}
# for i in range(max_number+1):
# 	for item in collection_labeled.find({"$and": [{'user_id':"user5"}, {'batch': i}]}):
# 		temp_data[item['batch']] = int(item['time'])/1000
# result = pd.DataFrame(data = {'batch' : list(temp_data.keys()), 'time': list(temp_data.values())})
# print(result)
# # batch_data = pd.DataFrame(data = [batch_data])
# batch_data = batch_data.T
# # batch_data.columns = ['batch', 'time']
# batch_data.rename(columns = {'': 'batch', 0 : 'time'}, inplace = True)
# batch_data.to_csv('./batch_data.csv')




# user_list = [1,2,3,6,8]
# positive_label = []
# negative_label = []
# # for i in user_list:

# positive_label += [item for item in collection_labeled.find({"$and": [{'user_id':"asdf"}, {'label': 1}]})]
# negative_label += [item for item in collection_labeled.find({"$and": [{'user_id':"asdf"}, {"$or": [{'label':0} , {'label':-1}]}]})]

# log_data = []
# log_data += [item for item in collection_log.find({"$and": [{'user_id':"user"+str(3)}, {'adjective': 'ATTRACTIVE'}]})]

# for i in range(10):
# 	batch_data = [item for item in colelction_labeled.find({"$and": [{'user_id':"user"+str(3)}, {'batch': str(i)}]})]
