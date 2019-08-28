import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from flask_pymongo import PyMongo
from sklearn.model_selection import train_test_split
from flask_pymongo import PyMongo
import pymongo
import pickle
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from sklearn.model_selection import cross_val_score

with open('./feature_label.pickle', 'rb') as f:
	data = pickle.load(f)

positive = []
negative = []

for item in data:
	if data[item][1] == 1:
		positive.append(data[item])
	else:
		negative.append(data[item])
negative = sample(negative, len(positive))

df_list = [item[0] for item in positive]
X1 = np.array(df_list)
Y1 = np.ones(X1.shape[0])

print(X1)
print(len(X1))

df_list = [item[0] for item in negative]
X2 = np.array(df_list)
Y2 = np.zeros(X2.shape[0])

print(X2)
print(len(X2))
X = np.concatenate((X1,X2), axis =0)
y = np.concatenate((Y1,Y2), axis =0)

classifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
classifier.fit(X_train, y_train)

result = classifier.score(X_test, y_test) 
predict_list = classifier.predict_proba(X_test)
scores = cross_val_score(classifier, X_test, y_test, cv=5)
print('prediction score: ', result)
print('CV score: ', scores)
# print('prediction list: ', predict_list)
