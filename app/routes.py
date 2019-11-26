from flask import render_template, request, url_for, jsonify, redirect, session
from flask_pymongo import PyMongo
from app import app
import pymongo
import os
import json
import pickle
import time
import random
import copy
from collections import Counter, OrderedDict
import ctypes
from datetime import datetime
import csv
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd 
import numpy as np


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

client = pymongo.MongoClient('mongodb://localhost:27017/') #for local
#client = pymongo.MongoClient("mongodb+srv://Dongjun:dongjun@labeling-kh64n.gcp.mongodb.net/test?retryWrites=true&w=majority")


db = client.davian
#-------------------------------Parameter---------------------------------------------------
CONST_BLUE_NUMBER = 0
CONST_RED_NUMBER = 0
CONST_NEUTRAL_NUMBER= 12
CONST_RANDOM_BLUE_NUMBER = 0
CONST_RANDOM_RED_NUMBER = 0
CONST_RANDOM_NEUTRAL_NUMBER= 12

CONST_BATCH_NUMBER = CONST_BLUE_NUMBER + CONST_NEUTRAL_NUMBER + CONST_RED_NUMBER
CONST_ADJECTIVE = ["EXTROVERTED", "CONFIDENTIAL","GOODNESS", "padding"]
# CONST_ADJECTIVE = ["YOUNG", "OVAL-FACED","BIG-NOSED", "padding"]
CONST_IMAGE_PATH = 'static/image/labeledEx/'
# CONST_IMAGE_PATH = 'static/image/FFHQ_SAMPLE2/FFHQ_SAMPLE2/'
CONST_PRETRAINED_FEATURE1 = "ffhq600_facenet_vggface1.pkl"
CONST_PRETRAINED_FEATURE2 = "ffhq600_facenet_vggface2.pkl"
#CONST_PRETRAINED_FEATURE3 = "attribute2_3.pkl"
CONST_CLUSTER_NUMBER = 200
CONST_CLUSTER_AFFINITY = "euclidean"
CONST_CLUSTER_LINKAGE = "ward"
CONST_SAMPLING_MODE = "GROUND"

#--------------------------------------------------------------------------------------------

# 처음 시작할 때, Vggface2 net으로 이미지의 feature를 추출한 뒤, pck 파일로 저장하여 활용합니다.

# from facenet_pytorch import InceptionResnetV1
# from PIL import Image
# from torchvision import transforms
# from tqdm import tqdm

# resnet = InceptionResnetV1(pretrained='vggface2').eval()

# image_names = os.listdir(os.path.join(APP_ROOT,CONST_IMAGE_PATH))
# features = {}
# for each_img_name in tqdm(image_names):
#     img = Image.open(os.path.join(os.path.join(APP_ROOT,CONST_IMAGE_PATH), each_img_name))
#     img = transforms.ToTensor()(img)
#     img_embedding = resnet(img.unsqueeze(0))
#     # print("img_embedding : {}".format(img_embedding.shape))
#     features[each_img_name] = img_embedding.squeeze(0).data.numpy()

# import pickle

# with open(CONST_FEATUREFILE_NAME, 'wb') as fp:
#     pickle.dump(features, fp)

#--------------------------------Function----------------------------------------------------
def read_pck(filename):
    objects = []
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
        return objects

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


def get_similar_images(input_data,feature_np,start,number, option):
    query_feature = None

    if option == "name":
        query_feature = np.expand_dims(np.array(features[input_data]), 0)
    elif option == "feature":
        query_feature = input_data
    print(query_feature.shape)
    print(feature_np.shape)
    ret = cosine_similarity(query_feature, feature_np)
    ret = np.squeeze(ret, 0)
    
    # print("ret shape",ret.shape, ret.shape[0])

    sort_ret = np.argsort(ret)[::-1][start:number+start]
    # print("sort_ret", sort_ret)
    return sort_ret

def get_attribute_images(attribute, attribute_temp, feature_temp, start, number, keyword_index):
    ret = []
    features = []
    index = []

    # 현재까지 한 user가 labeling 한 list (positive 혹은 negative)
    positive_label = [item for item in collection_labeled.find({"$and": [{'user_id':session.get("user_id")}, {'label': 1}, {"adjective" : CONST_ADJECTIVE[keyword_index]}]})]
    negative_label = [item for item in collection_labeled.find({"$and": [{'user_id':session.get("user_id")}, {'label':-1}, {"adjective" : CONST_ADJECTIVE[keyword_index]}]})]

    # positive를 아예 안할경우 대비 
    check_positive = False
    if len(positive_label) > 0:
        check_positive = True

    # image_id만 따로 뽑은 list
    positive_image_id_list = [item['image_id'] for item in positive_label]
    negative_image_id_list = [item['image_id'] for item in negative_label]

    # dictionary로 image_id랑 facial vector 연결
    positive_feature_vector = {}
    for image_id in positive_image_id_list:
        positive_feature_vector[image_id] = total_dict[image_id]
    negative_feature_vector = {}
    for image_id in negative_image_id_list:
        negative_feature_vector[image_id] = total_dict[image_id]

    # SVM에 들어갈 X,y 만듦 positive negative 섞음
    df_list = list(positive_feature_vector.values())
    X1 = np.array(df_list)
    Y1 = np.ones(X1.shape[0])
    df_list = list(negative_feature_vector.values())
    X2 = np.array(df_list)
    Y2 = np.zeros(X2.shape[0])
    print('X1', len(X1))
    print('X2', len(X2))


    if len(X1) == 0:
        X = X2
        y = Y2
    elif len(X2) == 0:
        X = X1
        y = Y1
    else:
        if X1.shape[0] > X2.shape[0]:
            X1 = random.sample(X1.tolist(),len(X2))
            Y1 = Y1[0:len(X2)]
            print('fitted length',len(X1), len(X2), len(Y1), len(Y2))
        else:
            X2 = random.sample(X2.tolist(),len(X1))
            Y2 = Y2[0:len(X1)]
            print('fitted length',len(X1), len(X2), len(Y1), len(Y2))

        X = np.concatenate((X1,X2), axis =0)
        y = np.concatenate((Y1,Y2), axis =0)

    classifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=True, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) -> 무시해도 됨

    # 한쪽이 0일 경우 그냥 통과 아니면 SVM

    isFitted = False
    if (len(X) == 0 and len (y) == 0) or check_positive == False:
        pass
    else:
        isFitted = True
        classifier.fit(X, y)
    
    for i in range(len(attribute_temp)):
        if attribute in attribute_temp[i]:
            features.append(feature_temp[i])
            index.append(i)
    
    if isFitted:
        if features:
            y_test = np.array(classifier.predict_proba(features)[:,0])    
            sort_ret = np.argsort(y_test)[::-1][0:number]
        else:
            sort_ret = []
    else:
        sort_ret = random.sample(range(len(attribute_temp)),number)

    ret = [index[item] for item in sort_ret]
    return ret

def appendImage(toList,possible_temp,Feature, query_indexes):
    for i in sorted(query_indexes):
        toList.append(possible_temp[i])    
    removeTemp(query_indexes,possible_temp,Feature)

def removeTemp(index,possible_temp,Feature):

    for i in sorted(index, reverse = True):
        del possible_temp[i]
        del Feature[i]
            
def removeFeature(Feature, labeledList):
    global total_image_list

    temp = []
    for item in labeledList:
        temp.append(total_image_list.index(item))


    for i in sorted(temp, reverse = True):
        del Feature[i]

    return np.array(Feature)

def extractCluster(data, cluster_data, option):    
    labeled_cluster = []
    for item in data:
        for i in range(len(cluster_data)):
            if str(item) in cluster_data[i]['image_list']:
                if option == 'index':
                    labeled_cluster.append(i)
                elif option == 'image_id':
                    labeled_cluster.append(cluster_data[i]['image_id'])
    return labeled_cluster


def choosingImage(data,adjective):
    posi_temp = []
    nega_temp = []
    neu_temp = []
    for item in data:
        if item['label']==1:
            posi_temp.append(item)
        elif item['label']==-1:
            nega_temp.append(item)
        else:
            neu_temp.append(item)
    if posi_temp:
        posi_name = posi_temp[random.randint(0,len(posi_temp)-1)]['image_id']
    else:
        posi_list = list(collection_labeled.find({"user_id":session.get("user_id"), "adjective":CONST_ADJECTIVE, "label":1}))
        if not posi_list:
            posi_list= list(collection_image.find())
        posi_name = posi_list[random.randint(0,len(posi_list)-1)]['image_id']
    if nega_temp:
        nega_name = nega_temp[random.randint(0,len(nega_temp)-1)]['image_id']
    else:
        nega_list = list(collection_labeled.find({"user_id":session.get("user_id"), "adjective":CONST_ADJECTIVE, "label":-1}))
        if not nega_list:
            nega_list = list(collection_image.find())
        nega_name = nega_list[random.randint(0,len(nega_list)-1)]['image_id']
    print(nega_name)
    return [posi_name, nega_name]

def updateLastcluster(data_list,total_cluster, option):
    sum_feature = None
    if option == "ajax":
        for i in range(len(data_list)):
            if i == 0:
                sum_feature = np.array(features[data_list[0]['image_id']])
            else:
                sum_feature += np.array(features[data_list[i]["image_id"]])
        sum_feature = np.expand_dims(sum_feature / CONST_BATCH_NUMBER,0)
    if option == "index":
        sum_feature = np.expand_dims(np.array(features[data_list]),0)

    cluster_feature = []
    for item in total_cluster:
        cluster_feature.append(features[item['image_id']])
    cluster_feature_np = np.array(cluster_feature)
    return total_cluster[get_similar_images(sum_feature,cluster_feature_np,0,1,"feature")[0]]['image_id']

def calculateAttribute(user_id, keyword_index, sampling_number):
    positive_labeled = list(collection_labeled.find({'user_id':user_id,'adjective':CONST_ADJECTIVE[keyword_index], "label":1, 'sampling':sampling_number},{'_id':0,'label':0,'user_id':0,'adjective':0,'time':0, 'sampling':0}))
    positive_labeled_list = [item['image_id'] for item in positive_labeled]
    negative_labeled = list(collection_labeled.find({'user_id':user_id,'adjective':CONST_ADJECTIVE[keyword_index], "label":-1, 'sampling':sampling_number},{'_id':0,'label':0,'user_id':0,'adjective':0,'time':0, 'sampling':0}))
    negative_labeled_list = [item['image_id'] for item in negative_labeled]

    del positive_labeled, negative_labeled

    ret_pos = []
    ret_neg = []
    if positive_labeled_list:
        positive_attribute = Counter({})
        for item in attr_list:
            if item in positive_labeled_list:
                positive_attribute = positive_attribute + Counter(attr_list[item])
    else:
        positive_attribute = Counter({})
    print(positive_attribute)
    if negative_labeled_list:
        negative_attribute = Counter({})
        for item in attr_list:
            if item in negative_labeled_list:
                negative_attribute = negative_attribute + Counter(attr_list[item])
    else:
        negative_attribute = Counter({})

    total_attribute = positive_attribute + negative_attribute
    
    for item in dict(positive_attribute):
        if total_attribute[item] < 10 or item == 'Attractive' or item == '5_o_Clock_Shadow':
            
            positive_attribute.pop(item, None)
        else:
            positive_attribute[item] /= total_attribute[item] 
        
    for item in dict(negative_attribute):
        if total_attribute[item] < 10 or item == 'Attractive' or item == '5_o_Clock_Shadow':
            negative_attribute.pop(item, None)
        else:
            negative_attribute[item] /= total_attribute[item]

    sorted_positive_score = dict(sorted(positive_attribute.items(), key=lambda x: x[1], reverse = True)[0:5])
    sorted_negative_score = dict(sorted(negative_attribute.items(), key=lambda x: x[1], reverse = True)[0:5])

    ret_pos = [{"attribute":item, "score":sorted_positive_score[item]} for item in sorted_positive_score]
    ret_neg = [{"attribute":item, "score":sorted_negative_score[item]} for item in sorted_negative_score] 
    return [ret_pos, ret_neg]


#-----------------------------------main----------------------------------------------------
db = client.davian

collection_labeled = db.labeled
collist = db.list_collection_names()

if "user" not in collist: #sampling count는 몇개나했는지 세는용도.
    collection_user = db.user
    collection_user.insert([{'_id':'asdf','pwd':'asdf','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0}, 
    {'_id':'labeler_1','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_2','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_3','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_4','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_5','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_6','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_7','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_8','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_9','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_10','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_11','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_12','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_13','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_14','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_15','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_16','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_17','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_18','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_19','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0},
    {'_id':'labeler_20','pwd':'davian','isDone':False, 'time': 0, 'sampling': 0, 'sampling_count': 0, 'latin': 0}])


collection_user = db.user
collection_log = db.log
collection_current = db.Current_toLabel
collection_cluster = db.cluster
collection_attribute = db.attribute

total_image_list1 = sorted(os.listdir(os.path.join(APP_ROOT,CONST_IMAGE_PATH)))
total_image_list2 = sorted(os.listdir(os.path.join(APP_ROOT,CONST_IMAGE_PATH)))


if "images" not in collist:
    collection_image = db.images

    temp = random.sample(range(len(total_image_list2)),750) ##2250
    collection_image.insert([{"image_id" : total_image_list2[i], "key_index" : 0} for i in temp])
    
    #for i in sorted(temp, reverse = True):
    #    del total_image_list2[i]
    print('total_image_list2:',len(total_image_list2))
    

    #temp = random.sample(range(len(total_image_list2)),750)
    #collection_image.insert([{"image_id" : total_image_list2[i], "key_index" : 1} for i in temp])
    
    #for i in sorted(temp, reverse = True):
    #    del total_image_list2[i]
    #print(len(total_image_list2))

    #temp = random.sample(range(len(total_image_list2)),750)
    #collection_image.insert([{"image_id" : total_image_list2[i], "key_index" : 2} for i in temp])
    
    #for i in sorted(temp, reverse = True):
    #    del total_image_list2[i]
    #print(len(total_image_list2))

    #temp = random.sample(range(len(total_image_list2)),750)
    #collection_image.insert([{"image_id" : total_image_list2[i], "key_index" : 3} for i in temp])

    print("check")

collection_image = db.images

total_num = 750 #2250 
flag_for_cluster = 0

features1 = read_pck(CONST_PRETRAINED_FEATURE1)[0]
features2 = read_pck(CONST_PRETRAINED_FEATURE2)[0]
#features1 = read_pck(CONST_PRETRAINED_FEATURE3)[0] #attractive feature

features1.update(features2)

#with open('./attribute2_3.pkl', 'rb') as f:     //for feature3 (attractive 만했던 이미지들 vgg face features)
#    total_dict = pickle.load(f)

with open('./ffhq600_facenet_vggface1.pkl', 'rb') as f:
    total_dict = pickle.load(f)
with open('./ffhq600_facenet_vggface2.pkl', 'rb') as f:
    dict2 = pickle.load(f)
total_dict.update(dict2)


features = {}
for key in sorted(features1.keys()):
    if not key in features:    # Depending on the goal, this line may not be neccessary
        features[key] = features1[key]

attr_list = {}
attr_list_temp = read_pck('attr_list.pickle')[0]
for key in sorted(attr_list_temp.keys()):
    if not key in attr_list:    # Depending on the goal, this line may not be neccessary
        attr_list[key] = attr_list_temp[key]
print(len(attr_list.keys()))

for j in range(0,1): #0,3 
    feature_list = []
    key_list = []
    attr_list2 = []

    total_image_list = sorted([item['image_id'] for item in list(collection_image.find({"key_index":j}))])
    total_num = len(total_image_list)
    print('total_num: ',total_num)
    features3 = {}
    for key in sorted(features1.keys()):
        if not key in features3 and key in total_image_list:    # Depending on the goal, this line may not be neccessary
            features3[key] = features1[key]
    print(len(features3.keys()))

    for each_key in sorted(features3):
        feature_list.append(features3[each_key]) #feature_list(feature vector sorted in key index)
        key_list.append(each_key)
    feature_np = np.array(feature_list)


    check = list(collection_cluster.find({'sampling':j}))
    if not check:
        # cluster = AgglomerativeClustering(n_clusters=CONST_CLUSTER_NUMBER, affinity=CONST_CLUSTER_AFFINITY, linkage=CONST_CLUSTER_LINKAGE).fit_predict(feature_list)
        cluster = AgglomerativeClustering(n_clusters=CONST_CLUSTER_NUMBER, affinity=CONST_CLUSTER_AFFINITY, linkage=CONST_CLUSTER_LINKAGE).fit_predict(feature_list)
        cluster = np.array(cluster)
        points = TSNE(n_components = 2, random_state=2019).fit_transform(feature_list)
            
        centeroid = []
        cluster_data = []

        for i in range(0,CONST_CLUSTER_NUMBER):
            arr = np.array([points[item] for item in np.where(cluster==i)])
            idx = np.array([item for item in np.where(cluster==i)])
            center = centeroidnp(arr[0])
            dist = [np.linalg.norm(arr[0][j] - center) for j in range(len(arr[0]))]   
            cluster_data.append({"sampling":j,
                                "image_id":key_list[idx[0][np.argmax(dist)]], 
                                "image_list": [str(item) for item in list(idx[0])],
                                "image_id_list": [total_image_list[item] for item in list(idx[0])],
                                "x":str(center[0]), 
                                "y":str(center[1]), 
                                "count":str(len(list(idx[0])))
                                })

        collection_cluster.insert(cluster_data)
        print(j, "inserted!")
    else:
        print("pass!")
client.close()
#---------------------------------------------------------------------------------------------------

#-------------------------------FRONT END - BACK END TRANSITION-------------------------------------

@app.route('/')
@app.route('/logIn', methods = ['GET','POST'])
def logIn():
    if request.method == 'GET':
        session.pop("logged_in",None)
        session.pop("user_id",None)
        return render_template('logIn.html')
    else:
        user_id = request.form['user']
        password = request.form['password']
        try:
            result = [item for item in collection_user.find({'_id': str(user_id), "pwd":str(password), "isDone":False})]
            if result:    
                
                collection_user.update({'_id':user_id}, {'$set':{'isLogOn' : True}})
                session['logged_in'] = True 
                session['user_id'] = user_id
                
                time = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                collection_log.insert({"Time":time,"user_id": user_id, "What":"Login"})

                return redirect(url_for('index'))
            else:
                return render_template('loginFail.html')
        except:
            return render_template('loginFail.html')

@app.route('/logout', methods = ['GET','POST'])

def logout():
    if request.method == 'GET':
        user_id = session.get("user_id")
        time = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        collection_log.insert({"Time":time,"user_id": user_id, "What":"Logout"})
        return redirect(url_for('logIn'))
        
@app.route('/getLog', methods = ['GET','POST'])
def getLog():
    if request.method == "POST":
        json_received = request.form
        data = json_received.to_dict(flat=False)
        data_list = json.loads(data['jsonData'][0])
        data_list['user_id'] = session.get('user_id')
        data_list['sampling'] = collection_user.find({'_id':session.get('user_id')})[0]['sampling']
        collection_log.insert(data_list)
        return jsonify("good")
        
@app.route('/getAttribute', methods = ['GET','POST'])
def getAttribute():
    if request.method == "POST":
        json_received = request.form
        data = json_received.to_dict(flat=False)
        data_list = json.loads(data['jsonData'][0])
        return jsonify({"attribute":attr_list[data_list['image_id']]})
        
@app.route('/getCurrent', methods = ['GET','POST'])
def getCurrent():
    blue_list = []
    red_list = []
    neutral_list = []

    user_id = session.get("user_id")
    if request.method == "POST":
        time = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

        json_received = request.form
        data = json_received.to_dict(flat=False)
        print(data)
        keyword_index = collection_current.find({"user_id": user_id})[0]['adjective']
        sampling_number = collection_user.find({'_id':user_id})[0]['sampling']

        db_image_list = sorted([item['image_id'] for item in collection_image.find({'key_index':sampling_number})])
        prelabeled_image_list = [item['image_id'] for item in collection_labeled.find({"user_id" : user_id, "adjective" : CONST_ADJECTIVE[keyword_index], 'key_index':sampling_number})]        
        possible_images = sorted(list(set(db_image_list) - set(prelabeled_image_list)))

        if data['type'][0] == "tsne":
            possible_temp = copy.deepcopy(possible_images)
            feature_temp = copy.deepcopy(feature_list)

            feature_removed = removeFeature(feature_temp, prelabeled_image_list)
 
            selectedImage = data['image_id'][0]
            collection_log.insert({"Time":time,"user_id": user_id, "What":"explore", "To":selectedImage})

            appendImage(blue_list, possible_temp, feature_temp, get_similar_images(selectedImage,feature_removed,0,CONST_BLUE_NUMBER, "name"))
            feature_removed = np.array(feature_temp)

            appendImage(red_list, possible_temp, feature_temp, get_similar_images(selectedImage,feature_removed,0,CONST_RED_NUMBER, "name"))
            feature_removed = np.array(feature_temp)

            appendImage(neutral_list, possible_temp, feature_temp, get_similar_images(selectedImage,feature_removed,0,CONST_NEUTRAL_NUMBER, "name"))
        
        elif data['type'][0] == "attribute":
            possible_temp = copy.deepcopy(possible_images)
            attribute_dict = copy.deepcopy(attr_list)
            feature_dict = copy.deepcopy(features)

            cluster_temp = list(collection_cluster.find({'sampling':sampling_number}))
        
            feature_temp = []
            for item in possible_temp:
                feature_temp.append(feature_dict[item])

            attribute_temp = []
            for item in possible_temp:
                attribute_temp.append(attribute_dict[item])

            selectedAttribute = data['attribute'][0]

            collection_log.insert({"Time":time,"user_id": user_id, "What":"explore", "To":selectedAttribute, "sampling":sampling_number})

            appendImage(neutral_list, possible_temp, attribute_temp, get_attribute_images(selectedAttribute, attribute_temp, feature_temp, 0,CONST_NEUTRAL_NUMBER, keyword_index))

        current_todo = blue_list + neutral_list + red_list

        labeled = [db_image_list.index(item) for item in current_todo]        
        cluster = extractCluster(labeled, cluster_temp, "image_id")
        print(cluster)

        for i in range(CONST_BATCH_NUMBER):
            if len(current_todo) > i:
                collection_current.update({"user_id":user_id , "index":i}, {"user_id":user_id , "index":i, "adjective": keyword_index, "image_id" : current_todo[i]})
            else:
                collection_current.update({"user_id":user_id , "index":i}, {"user_id":user_id , "index":i, "adjective": keyword_index, "image_id" : None})
        
        attribute_score = calculateAttribute(user_id, keyword_index, sampling_number)

    return jsonify({"blue":blue_list, "neutral":neutral_list, "red": red_list,
                     "keyword": CONST_ADJECTIVE[keyword_index],
                    "image_count" : (int((total_num - len(possible_images))/CONST_BATCH_NUMBER)+1), 
                    "index": keyword_index,
                    "isNewset" : False,
                    "score" : [],
                    "current_cluster" : cluster,
                    "positive_attr_list" : attribute_score[0],
                    "negative_attr_list" : attribute_score[1]})

@app.route('/getData', methods = ['GET','POST'])
def getData():
    # cumulative_result = {}
    # cumulative_data = []
    user_id = session.get("user_id")
    # cumulative_result['user_id'] = user_id
    check_time = collection_user.find({'_id':user_id})[0]['time']
    print('check_time', check_time)
    #data 추가하는 것 try except 문으로 또 걸어주기 (id, pwd)까지
    if request.method == "POST":
        blue_number = 0
        red_number = 0
        neutral_number = 0
        user = collection_user.find({'_id':user_id})[0]
        sampling_number = user['sampling']
        sampling_count = user['sampling_count']
        latin = user['latin']

        blue_list = []
        red_list = []
        neutral_list = []
        batch_list = []
        isNewset = None
        time = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        json_received = request.form
        data = json_received.to_dict(flat=False)
        data_list = json.loads(data['jsonData'][0])
        
        keyword_index = collection_current.find({"user_id": user_id})[0]['adjective']
        print("keyword_index", keyword_index)

        cluster_temp = list(collection_cluster.find({'sampling':(sampling_number+latin)%3},{'_id':0}))
        total_image_temp = sorted([item['image_id'] for item in list(collection_image.find({'key_index':(sampling_number+latin)%3}))])
        print(sampling_number)

        # 현재까지 한 user가 labeling 한 list (positive 혹은 negative)
        positive_label = [item for item in collection_labeled.find({"$and": [{'user_id':user_id}, {'label': 1}, {"adjective" : CONST_ADJECTIVE[keyword_index]}, {"sampling":sampling_number}]})]
        negative_label = [item for item in collection_labeled.find({"$and": [{'user_id':user_id}, {'label':-1}, {"adjective" : CONST_ADJECTIVE[keyword_index]}, {"sampling":sampling_number}]})]

        #현재 labeling 한것을 추가
        for item in data_list:
            if item['label'] == 1 and item not in positive_label:
                positive_label.append(item)
            elif item['label'] == -1 and item not in negative_label:
                negative_label.append(item)

        # image_id만 따로 뽑은 list
        positive_image_id_list = [item['image_id'] for item in positive_label]
        negative_image_id_list = [item['image_id'] for item in negative_label]

        # dictionary로 image_id랑 facial vector 연결
        positive_feature_vector = {}
        for image_id in positive_image_id_list:
            positive_feature_vector[image_id] = total_dict[image_id]
        negative_feature_vector = {}
        for image_id in negative_image_id_list:
            negative_feature_vector[image_id] = total_dict[image_id]

        # SVM에 들어갈 X,y 만듦 positive negative 섞음
        df_list = list(positive_feature_vector.values())
        X1 = np.array(df_list)
        Y1 = np.ones(X1.shape[0])
        df_list = list(negative_feature_vector.values())
        X2 = np.array(df_list)
        Y2 = np.zeros(X2.shape[0])

        print('X1', len(X1))
        print('X2', len(X2))

        if len(X1) == 0:
            X = X2
            y = Y2
        elif len(X2) == 0:
            X = X1
            y = Y1
        else:
            
            if X1.shape[0] > X2.shape[0]:
                X1 = random.sample(X1.tolist(),len(X2))
                Y1 = Y1[0:len(X2)]
                print('fitted length',len(X1), len(X2), len(Y1), len(Y2))
            else:
                X2 = random.sample(X2.tolist(),len(X1))
                Y2 = Y2[0:len(X1)]
                print('fitted length',len(X1), len(X2), len(Y1), len(Y2))

            X = np.concatenate((X1,X2), axis =0)
            y = np.concatenate((Y1,Y2), axis =0)

        classifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=True, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) -> 무시해도 됨

        # 한쪽이 0일 경우 그냥 통과 아니면 SVM

        isFitted = False
        if (np.sum(y) == 0 or np.sum(y)%12 == 0):
            pass
        else:
            isFitted = True
            classifier.fit(X, y)

        before_time = collection_user.find({'_id':user_id})[0]['time']
        if data_list:
            add_time = int(data_list[0]['time'])
        else:
            add_time = 0
        final_time = before_time + add_time

        print('final_time', final_time)
        
        collection_user.update({'_id':user_id}, {'$set':{'time' : final_time}})

        for item in data_list:
            item['user_id'] = user_id
            item['sampling'] = sampling_number

        collection_log.insert({"Time":time,"user_id": user_id, "What":"confirm", "sampling": sampling_number})
        
        if data_list:
            for item in data_list:
                check = list(collection_labeled.find({"user_id": item["user_id"], "adjective": item["adjective"], "image_id": item['image_id'], "sampling" : sampling_number}))
                if check:
                    collection_labeled.update({"user_id": item["user_id"], "adjective": item["adjective"], "image_id": item['image_id'], "sampling" : sampling_number}, 
                                            {'$set': {"user_id": item["user_id"],"cluster": item["cluster"], "image_id": item['image_id']
                                                        , "adjective": item["adjective"], "label":item["label"], "time":item['time'], "sampling" : sampling_number}})
                    print("updated!")
                else:
                    collection_labeled.insert(item)
                    print("inserted!")
      
        keyword_index = collection_current.find({"user_id": user_id})[0]['adjective']
        
        imageStandard = choosingImage(data_list, CONST_ADJECTIVE[keyword_index])

        db_image_list = [item['image_id'] for item in collection_image.find({"key_index" : (sampling_number + latin)%3})]
        print(len(db_image_list))

        prelabeled_image_list = [item['image_id'] for item in collection_labeled.find({"user_id" : user_id, "adjective" : CONST_ADJECTIVE[keyword_index], "sampling" : sampling_number})]        
        possible_images = sorted(list(set(db_image_list) - set(prelabeled_image_list)))


        if not possible_images:
            
            if sampling_count == 2:
                collection_user.update({'_id':user_id}, {'$set':{'isDone' : True}})
            else:      
                collection_user.update({'_id':user_id}, {'$set':{'sampling' : (sampling_number + 1)%3, 'time' : 0, 'sampling_count': sampling_count+1}})

            if sampling_number == 0:
                blue_number = CONST_RANDOM_BLUE_NUMBER
                red_number = CONST_RANDOM_RED_NUMBER
                neutral_number = CONST_RANDOM_NEUTRAL_NUMBER
                isNewset = True
                keyword_index = keyword_index + 1

                db_image_list = [item['image_id'] for item in collection_image.find({'key_index':(0 + latin)%3})]
                prelabeled_image_list = [item['image_id'] for item in collection_labeled.find({"user_id" : user_id, "adjective" : CONST_ADJECTIVE[keyword_index]})]        
                possible_images = sorted(list(set(db_image_list) - set(prelabeled_image_list)))
                
                possible_temp = copy.deepcopy(possible_images)
                print(len(possible_temp))
                feature_temp = copy.deepcopy(feature_list)
                

                appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),CONST_BATCH_NUMBER))
                batch_list = neutral_list

            elif sampling_number == 2:
                blue_number = CONST_RANDOM_BLUE_NUMBER
                red_number = CONST_RANDOM_RED_NUMBER
                neutral_number = CONST_RANDOM_NEUTRAL_NUMBER
                isNewset = True
                keyword_index = keyword_index + 1
                
                db_image_list = [item['image_id'] for item in collection_image.find({"key_index" : (0 + latin)%3})] #2 + latin
                print(len(db_image_list))
            
                prelabeled_image_list = [item['image_id'] for item in collection_labeled.find({"user_id" : user_id, "adjective" : CONST_ADJECTIVE[keyword_index]})]        
                possible_images = sorted(list(set(db_image_list) - set(prelabeled_image_list)))
                
                possible_temp = copy.deepcopy(possible_images)
                print(len(possible_temp))
                feature_temp = copy.deepcopy(feature_list)
                

                appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),CONST_BATCH_NUMBER))
                batch_list = neutral_list


            elif sampling_number == 1:
                blue_number = CONST_RANDOM_BLUE_NUMBER
                red_number = CONST_RANDOM_RED_NUMBER
                neutral_number = CONST_RANDOM_NEUTRAL_NUMBER
                isNewset = True
                keyword_index = keyword_index + 1
                
                db_image_list = [item['image_id'] for item in collection_image.find({"key_index" : (1 + latin)%3})]
                print(len(db_image_list))
            
                prelabeled_image_list = [item['image_id'] for item in collection_labeled.find({"user_id" : user_id, "adjective" : CONST_ADJECTIVE[keyword_index]})]        
                possible_images = sorted(list(set(db_image_list) - set(prelabeled_image_list)))
                
                possible_temp = copy.deepcopy(possible_images)
                print(len(possible_temp))
                feature_temp = copy.deepcopy(feature_list)
                

                appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),CONST_BATCH_NUMBER))
                batch_list = neutral_list

        else:
            isNewset = False
            print("possible_images", len(possible_images))
            possible_temp = copy.deepcopy(possible_images)
            feature_dict = copy.deepcopy(features)
            feature_temp = []
            for item in possible_temp:
                feature_temp.append(feature_dict[item])
            
            print("feature_removed", len(feature_temp))

            # 여기서 모델로 사진 결정
            global flag_for_cluster

            if sampling_number == 2:
                print('RANDOM')
                if len(possible_temp) >= CONST_RANDOM_NEUTRAL_NUMBER:
                    appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),CONST_RANDOM_NEUTRAL_NUMBER))
                else:
                    appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),len(possible_temp)))
                blue_number = CONST_RANDOM_BLUE_NUMBER
                red_number = CONST_RANDOM_RED_NUMBER
                neutral_number = CONST_RANDOM_NEUTRAL_NUMBER

                batch_list = neutral_list

            elif sampling_number == 0: ##sampling_number 0, latin 0 일때 현재 오류가 안남.
                print('CLUSTER')
                if isFitted:
                    if flag_for_cluster == 0:
                        y_test = np.array(classifier.predict_proba(feature_temp)[:,0])
                        max_ret = np.argsort(y_test)[::-1][0:3]
                        #print('feature:', feature_temp[max_ret[0]], len(feature_temp[max_ret[0]])) #512
                        print("flag:",flag_for_cluster)
                        print('max_ret:', max_ret)

                        global max_selected_image1

                        max_selected_image1 = possible_temp[max_ret[0]]
                        
                        print('Max selected image :', max_selected_image1)
                        
                        
                        similar_images = get_similar_images(max_selected_image1, np.array(feature_temp),0,CONST_NEUTRAL_NUMBER, "name")
                        similar_images1 = similar_images
                        
                        
                        #similar_images2 = get_similar_images(max_selected_image2, np.array(feature_temp),0,CONST_NEUTRAL_NUMBER, "name")
                        #similar_images3 = get_similar_images(max_selected_image3, np.array(feature_temp),0,CONST_NEUTRAL_NUMBER, "name")
                        #print("Similar images : ", similar_images1, similar_images2,similar_images3)
                        flag_for_cluster = 1
                        print("changed_flag:",flag_for_cluster)
                        print('possible_temp :',len(possible_temp))
                        print('feature_temp:', len(feature_temp))
                        appendImage(neutral_list, possible_temp, feature_temp, similar_images1)
                        
                        
                        
                    elif flag_for_cluster == 1:
                        similar_images = get_similar_images(max_selected_image1, np.array(feature_temp),0,CONST_NEUTRAL_NUMBER, "name")
                        similar_images2 = similar_images
                        print(similar_images2)
                        print("flag:", flag_for_cluster)
                        print('possible_temp :',len(possible_temp))
                        print('feature_temp:', len(feature_temp))
                        flag_for_cluster = 2
                        appendImage(neutral_list, possible_temp, feature_temp, similar_images2)
                        

                    elif flag_for_cluster == 2:
                        similar_images = get_similar_images(max_selected_image1, np.array(feature_temp),0,CONST_NEUTRAL_NUMBER, "name")
                        similar_images3 = similar_images
                        print("flag:", flag_for_cluster)
                        print('possible_temp :',len(possible_temp))
                        print('feature_temp:', len(feature_temp))
                        flag_for_cluster = 0
                        appendImage(neutral_list, possible_temp, feature_temp, similar_images3)
                        
                else:
                    if len(possible_temp) >= CONST_RANDOM_NEUTRAL_NUMBER:
                        appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),CONST_RANDOM_NEUTRAL_NUMBER))
                    else:
                        appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),len(possible_temp)))

                blue_number = CONST_RANDOM_BLUE_NUMBER
                red_number = CONST_RANDOM_RED_NUMBER
                neutral_number = CONST_RANDOM_NEUTRAL_NUMBER

                batch_list = neutral_list

            elif sampling_number == 3:
                print('TOP')
                if isFitted:
                    y_test = np.array(classifier.predict_proba(feature_temp)[:,0])    
                    max_ret = np.argsort(y_test)[::-1][0:1][0]
                    print(max_ret)
                    max_selected_image = possible_temp[max_ret]
                    print(max_selected_image)
                    
                    #print(prelabeled_image_list) #labeled 된 이미지 id이름들
                    
                    #sort_ret = np.argsort(y_test)[::-1][0:CONST_BATCH_NUMBER]
                    print(len(feature_temp))
                    #print(sort_ret)            
                    #appendImage(neutral_list, possible_temp, feature_temp, sort_ret)
                    appendImage(neutral_list, possible_temp, feature_temp, get_similar_images(max_selected_image,np.array(feature_temp),0,CONST_NEUTRAL_NUMBER, "name"))

                else:
                    if len(possible_temp) >= CONST_RANDOM_NEUTRAL_NUMBER:
                        appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),CONST_RANDOM_NEUTRAL_NUMBER))
                    else:
                        appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),len(possible_temp)))

                blue_number = CONST_RANDOM_BLUE_NUMBER
                red_number = CONST_RANDOM_RED_NUMBER
                neutral_number = CONST_RANDOM_NEUTRAL_NUMBER

                batch_list = neutral_list

            elif sampling_number == 1:
                print('MIDDLE')
                print(isFitted)
                if isFitted:
                    y_test = np.array(classifier.predict_proba(feature_temp)[:,0])    
                    sort_ret = np.argsort(abs(y_test-0.5))[::1][0:CONST_BATCH_NUMBER]
                    appendImage(neutral_list, possible_temp, feature_temp, sort_ret)

                else:
                    if len(possible_temp) >= CONST_RANDOM_NEUTRAL_NUMBER:
                        appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),CONST_RANDOM_NEUTRAL_NUMBER))
                    else:
                        appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),len(possible_temp)))

                blue_number = CONST_RANDOM_BLUE_NUMBER
                red_number = CONST_RANDOM_RED_NUMBER
                neutral_number = CONST_RANDOM_NEUTRAL_NUMBER

                batch_list = neutral_list

        current_todo = batch_list #해야하는 것들 
            

        labeled = [total_image_temp.index(item['image_id']) for item in data_list]
        current_cluster_index = [total_image_temp.index(item) for item in current_todo]
        current_cluster = extractCluster(current_cluster_index, cluster_temp, "image_id")
        print('current_cluster:', current_cluster)
        labeled_cluster = extractCluster(labeled, cluster_temp, 'image_id')
        label = np.array([d['label'] for d in data_list])

        for i in range(0,CONST_BATCH_NUMBER):
            if len(current_todo) > i:
                collection_current.update({"user_id":user_id , "index":i}, {"user_id":user_id , "index":i, "adjective": keyword_index, "image_id" : current_todo[i], 'sampling' : sampling_number})
            else:
                collection_current.update({"user_id":user_id , "index":i}, {"user_id":user_id , "index":i, "adjective": keyword_index, "image_id" : None, 'sampling' : sampling_number})
        if keyword_index >= 3:
            collection_user.update({'_id':user_id}, {'$set':{'isDone' : True}})


        outScore = []
        temp = []
        for i in range(len(labeled_cluster)):
            if labeled_cluster[i] not in temp:
                temp.append(labeled_cluster[i])
                outScore.append({"image_id" : labeled_cluster[i], "score" : label[i], "labeled" : True})
            else:
                outScore[[d['image_id'] for d in outScore].index(labeled_cluster[i])]["score"] += label[i]
        
        for i in range(len(outScore)):
            count = int(cluster_temp[[item['image_id'] for item in cluster_temp].index(outScore[i]["image_id"])]['count'])
            outScore[i]['score'] = outScore[i]['score'] / count

        attribute_score = calculateAttribute(user_id, keyword_index, sampling_number)
        print("blue", current_todo[0:CONST_BLUE_NUMBER])
        return jsonify({"blue":current_todo[0:blue_number], "neutral":current_todo[blue_number:blue_number+neutral_number], "red": current_todo[blue_number+neutral_number:blue_number+neutral_number+red_number],
                     "keyword": CONST_ADJECTIVE[keyword_index],
                    "image_count" : (int((total_num - len(possible_images))/CONST_BATCH_NUMBER)+1), 
                    "index": keyword_index,
                    "isNewset" : isNewset,
                    "score" : outScore,
                    "current_cluster" : current_cluster,
                    "positive_attr_list" : attribute_score[0],
                    "negative_attr_list" : attribute_score[1],
                    "time" : final_time})

    

@app.route('/index', methods = ['GET', 'POST'])
def index():

    blue_list = []
    red_list = []
    neutral_list = []
    
    print(session.get('logged_in'))
    if session.get("logged_in")==True:
        print(session.get("user_id"), session.get("logged_in"))
        
        user_id = session.get("user_id")
        user = collection_user.find({'_id':user_id})[0]
        sampling_number = user['sampling']
        latin = user['latin']

        db_image_list = [item['image_id'] for item in collection_image.find({'key_index':(0 + latin)%3})] #key_index : sampling + latin
        print(len(db_image_list))
        todo_images = [item for item in collection_current.find({"user_id" : user_id, 'sampling':sampling_number})] #해야되는 이미지들 
        keyword_index = 0
        if todo_images:
            keyword_index = todo_images[-1]["adjective"]

        print(todo_images)
        if todo_images:
            print("old")
            dictOfImg = { i : todo_images[i]['image_id'] for i in range(0,CONST_BATCH_NUMBER)}
            
        else:        
            print("new")
            if sampling_number == 0:
                batch_list = [db_image_list[item] for item in random.sample(range(len(db_image_list)),CONST_BATCH_NUMBER)]
            
            elif sampling_number == 2:
                batch_list = [db_image_list[item] for item in random.sample(range(len(db_image_list)),CONST_BATCH_NUMBER)]

            elif sampling_number == 1:
                batch_list = [db_image_list[item] for item in random.sample(range(len(db_image_list)),CONST_BATCH_NUMBER)]

        
            
            dictOfImg = { i : batch_list[i] for i in range(0,CONST_BATCH_NUMBER)}
            keyword_index = 0
            collection_current.delete_many({'user_id':user_id})
            collection_current.insert([{'user_id' : user_id, "adjective" : 0, 'index' : i, 'image_id' : dictOfImg[i], "sampling" : sampling_number} for i in range(0,CONST_BATCH_NUMBER)])

        cluster_temp = list(collection_cluster.find({'sampling':(0 + latin)%3},{'_id':0})) #sampling_number
        print("cluster_temp", len(cluster_temp), cluster_temp[0])

        total_image_temp = sorted([item['image_id'] for item in list(collection_image.find({'key_index':(0+latin)%3}))]) #0 -> sampling number
        current_cluster_index = []
        for item in dictOfImg.keys():
            if dictOfImg[item] != None:
                current_cluster_index.append(total_image_temp.index(dictOfImg[item]))
        current_cluster = extractCluster(current_cluster_index, cluster_temp,"image_id")

        labeled_data = list(collection_labeled.find({'user_id':user_id,'adjective':CONST_ADJECTIVE[keyword_index], 'sampling':sampling_number},{'_id':0,'user_id':0,'adjective':0,'time':0}))
        labeled = [total_image_temp.index(item['image_id']) for item in labeled_data]
        labeled_cluster = extractCluster(labeled, cluster_temp, 'index')

        for i in range(0,CONST_CLUSTER_NUMBER):
            sum_score = 0
            for j in range(len(labeled_cluster)):
                if labeled_cluster[j] == i:
                    sum_score += labeled_data[j]['label']
                    cluster_temp[i]['labeled'] = True
            cluster_temp[i]['score'] = str(sum_score/len(cluster_temp[i]['image_list']))

        count = int(len(labeled_data)/CONST_BATCH_NUMBER)+1
        
        # 여기서 첫 세트 사진 결정
        # 형용사 결정

        images = json.dumps(dictOfImg)
        label = json.dumps(labeled_data)
        current_cluster_json = json.dumps(current_cluster)
        cluster_json = json.dumps(cluster_temp)

        attribute_score = calculateAttribute(user_id, keyword_index, sampling_number)

        positive_score = json.dumps(attribute_score[0])
        negative_score = json.dumps(attribute_score[1])

        return render_template('photolabeling.html', keyword = CONST_ADJECTIVE[keyword_index],
                                                     images = images, user_id = user_id, 
                                                     total_num = int(total_num/CONST_BATCH_NUMBER)+1, 
                                                     count_num = count,
                                                     label = label,
                                                     positive_attr_list = positive_score,
                                                     negative_attr_list = negative_score,
                                                     cluster = cluster_json,
                                                     current_cluster = current_cluster_json
                                                     )
    
    else:
        return redirect(url_for('logIn'))
