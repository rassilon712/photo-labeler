from flask_pymongo import PyMongo
import pymongo
import pickle
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


client = pymongo.MongoClient("mongodb+srv://admin:davian@daviandb-9rvqg.gcp.mongodb.net/test?retryWrites=true&w=majority")
# client = pymongo.MongoClient('mongodb://localhost:27017/')

# ------------------------------------------- parameter settings -----------------------------
db = client.davian
collection_labeled = db.labeled
collection_log = db.log
user_list = [6]
total_data_time_per_batch = {}
total_data_time_per_batch_df = pd.DataFrame()
total_data_count_number = {}
total_data_count_number_df = pd.DataFrame()
total_data_click_per_batch = {}
total_data_click_per_batch_df = pd.DataFrame()

# ---------------------------------------------------------------------------------------------

def maxNumber(data):
	temp = [item['batch'] for item in data]
	max_number = temp[len(temp)-1]
	return max_number

def countNumber(data, user):
	positive_labeling = []
	negative_labeling = []
	for item in data:
		if item['label'] == 1:
			positive_labeling.append(item)
		else:
			negative_labeling.append(item)
	assert len(data) == len(positive_labeling) + len(negative_labeling)
	total_data_count_number[user] = [len(positive_labeling), len(negative_labeling), len(data)]

def clickPerBatch(data, max_number, user):
	temp_data = []
	for batch in range(1, max_number+1):
		cnt = 0
		for item in data:
			if item['batch'] == batch:
				cnt += 1
		temp_data.append(cnt)
		total_data_click_per_batch[user] = temp_data 

def timePerBatch(data, max_number, user):
	temp_data = {}
	for i in range(max_number+1):
		for item in collection_labeled.find({"$and": [{'user_id':user}, {'batch': i}]}):
			temp_data[item['batch']] = int(item['time'])/1000
	result = pd.DataFrame(data = {'batch' : list(temp_data.keys()), 'time': list(temp_data.values())})
	total_data_time_per_batch[user] = list(temp_data.values())
	x = result['batch']
	y = result['time']
	plt.title(user + " time per batch")
	plt.legend(loc='upper left', frameon=True)
	plt.ylabel('time')
	plt.xlabel('batch')
	plt.plot(x,y, label=user)
	plt.show()


def calculateTotal(data):
	data_df = pd.DataFrame()
	for item in data:
		temp_df = pd.DataFrame(data = {item : data[item]})
		data_df = pd.concat([data_df, temp_df], axis=1)
	if data == total_data_count_number:
		data_df = data_df.rename(index = {0: 'positive_labeling', 1: 'negative_labeling', 2: 'total_labeling'})
	else:
		pass
	return data_df


# --------------------------------------- main part -----------------------------------------

total_data = []
for user in user_list:
	user = "user"+str(user)
	user_data = [item for item in collection_labeled.find({"$and": [{'user_id':user}]})]

	log_data = []
	for item in collection_log.find({"$and": [{'user_id':user}]}):
		if len(item) > 5:
			log_data.append(item)

	max_number = maxNumber(user_data)
	max_number_log = maxNumber(log_data)		
	timePerBatch(user_data, max_number, user)	
	countNumber(user_data,user)
	clickPerBatch(log_data,max_number_log,user)
	total_data.extend(user_data)


#  ----------------------------------------- calculate total -------------------------------- 
total_data_count_number_df = calculateTotal(total_data_count_number)
total_data_time_per_batch_df = calculateTotal(total_data_time_per_batch)
total_data_click_per_batch_df = calculateTotal(total_data_click_per_batch)

# ----------------------------------------- print all measurements --------------------------
print('count number: ')
print(total_data_count_number_df)
print('time per batch: ')
print(total_data_time_per_batch_df)
print('click per batch: ')
print(total_data_click_per_batch_df)
