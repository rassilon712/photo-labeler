import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from flask_pymongo import PyMongo
from sklearn.model_selection import train_test_split
import pymongo
import pickle
import pandas as pd 
client = pymongo.MongoClient("mongodb+srv://admin:davian@daviandb-9rvqg.gcp.mongodb.net/test?retryWrites=true&w=majority")
db = client.davian
collection_labeled = db.labeled
collection_image = db.images
user_list = [1,2,3,6,8]
positive_label = []
negative_label = []
for i in user_list:
	positive_label += [item for item in collection_labeled.find({"$and": [{'user_id':"user"+str(i)}, {'label': 1}]})]
	negative_label += [item for item in collection_labeled.find({"$and": [{'user_id':"user"+str(i)}, {"$or": [{'label':0} , {'label':-1}]}]})]
with open('./ffhq600_facenet_vggface1.pkl', 'rb') as f:
	total_dict = pickle.load(f)
with open('./ffhq600_facenet_vggface2.pkl', 'rb') as f:
	dict2 = pickle.load(f)
total_dict.update(dict2)
positive_image_id_list = [item['image_id'] for item in positive_label]
negative_image_id_list = [item['image_id'] for item in negative_label]

positive_feature_vector = {}
for image_id in positive_image_id_list:
	positive_feature_vector[image_id] = (total_dict[image_id], 1)

negative_feature_vector = {}
for image_id in negative_image_id_list:
	negative_feature_vector[image_id] = (total_dict[image_id], -1)

# final_dict = {}
# final_dict.update(positive_feature_vector)
# final_dict.update(negative_feature_vector)

# print(len(final_dict))
# with open('feature_label.pickle', 'wb') as f:
# 	pickle.dump(final_dict, f)

df_list = list(positive_feature_vector.values())
X1 = np.array(df_list)
Y1 = np.ones(X1.shape[0])

df_list = list(negative_feature_vector.values())
X2 = np.array(df_list)
Y2 = np.zeros(X2.shape[0])

X = np.concatenate((X1,X2), axis =0)
y = np.concatenate((Y1,Y2), axis =0)

classifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
classifier.fit(X_train, y_train)

train_dict = {}
for i in range(len(X_test)):
	for key, value in total_dict.items():
		if X_test[i][0] == value[0] and X_test[i][1] == value[1] and X_test[i][2] == value[2] and X_test[i][3] == value[3] and X_test[i][4] == value[4]:
			train_dict[key] = value
		# else:
		# 	print(X_test)
# print(train_dict)
# print(len(X_test))
# print(len(train_dict.keys()))
result = classifier.score(X_test, y_test) 
predict_list = classifier.predict_proba(X_test)
print('prediction score: ', result)
print('prediction list: ', predict_list)
df = pd.DataFrame(result, index= train_dict.keys())
print(df)
print(classifier.predict_proba(X_test))
print(type(result))