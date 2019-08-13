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

from datetime import datetime
import csv

from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd 
import numpy as np


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

client = pymongo.MongoClient('mongodb://localhost:27017/')
# client = pymongo.MongoClient("mongodb+srv://admin:davian@daviandb-9rvqg.gcp.mongodb.net/test?retryWrites=true&w=majority")


db = client.davian
#-------------------------------Parameter---------------------------------------------------
CONST_BLUE_NUMBER = 6
CONST_RED_NUMBER = 6
CONST_NEUTRAL_NUMBER= 2
CONST_RANDOM_BLUE_NUMBER = 0
CONST_RANDOM_RED_NUMBER = 0
CONST_RANDOM_NEUTRAL_NUMBER= 14

CONST_BATCH_NUMBER = CONST_BLUE_NUMBER + CONST_NEUTRAL_NUMBER + CONST_RED_NUMBER
CONST_ADJECTIVE = ["ATTRACTIVE", "CONFIDENTIAL","GOODNESS", "padding"]
CONST_IMAGE_PATH = 'static/image/FFHQ_SAMPLE2'
CONST_PRETRAINED_FEATURE = "ffhq600_facenet_vggface2.pkl"
CONST_CLUSTER_NUMBER = 200
CONST_CLUSTER_AFFINITY = "euclidean"
CONST_CLUSTER_LINKAGE = "ward"
CONST_FEATUREFILE_NAME = 'ffhq600_facenet_vggface2.pkl'
CONST_SAMPLING_MODE = "CLUSTER"

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

def get_attribute_images(attribute,attribute_list,start,number):
    ret = []
    
    for i in range(len(attribute_list)):
        if attribute in attribute_list[i]:
            ret.append(i)
            print(attribute_list[i], attribute)            
        if len(ret) >= number:
            break
    return ret


def get_not_attribute_images(attribute,attribute_list,start,number):
    ret = []
    
    for i in range(len(attribute_list)):
        if attribute not in attribute_list[i]:
            ret.append(i)
            print(attribute_list[i], attribute)            
        if len(ret) >= number:
            break
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

def extractCluster(data, option):    
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

def calculateAttribute(user_id, keyword_index):
    positive_labeled = list(collection_labeled.find({'user_id':user_id,'adjective':CONST_ADJECTIVE[keyword_index], "label":1},{'_id':0,'label':0,'user_id':0,'adjective':0,'time':0}))
    positive_labeled_list = [item['image_id'] for item in positive_labeled]
    negative_labeled = list(collection_labeled.find({'user_id':user_id,'adjective':CONST_ADJECTIVE[keyword_index], "label":-1},{'_id':0,'label':0,'user_id':0,'adjective':0,'time':0}))
    negative_labeled_list = [item['image_id'] for item in negative_labeled]

    del positive_labeled, negative_labeled

    ret_pos = []
    ret_neg = []
    if positive_labeled_list:
        positive_attribute = Counter({})
        for item in attr_list:
            if item in positive_labeled_list:
                positive_attribute = positive_attribute + Counter(attr_list[item])
        # for item in positive_attribute:
        #     positive_attribute[item] /= len(positive_labeled_list)
    else:
        positive_attribute = Counter({})
    print(positive_attribute)
    if negative_labeled_list:
        negative_attribute = Counter({})
        for item in attr_list:
            if item in negative_labeled_list:
                negative_attribute = negative_attribute + Counter(attr_list[item])
        # for item in negative_attribute:
        #     negative_attribute[item] /= len(negative_labeled_list)
    else:
        negative_attribute = Counter({})

    total_attribute = positive_attribute + negative_attribute
    
    for item in dict(positive_attribute):
        if total_attribute[item] < 10:
            positive_attribute.pop(item, None)
        else:
            positive_attribute[item] /= total_attribute[item] 
        
    for item in dict(negative_attribute):
        if total_attribute[item] < 10:
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
if "images" in collist:
    print("check")
    db.images.drop()
if "user" in collist:
    db.user.drop()

collection_user = db.user

collection_user.insert([{'_id':'asdf','pwd':'asdf','isDone':False}, {'_id':'user101','pwd':'davian101','isDone':False},{'_id':'user1','pwd':'davian','isDone':False},{'_id':'user2','pwd':'davian','isDone':False},{'_id':'user3','pwd':'davian','isDone':False},{'_id':'user4','pwd':'davian','isDone':False},{'_id':'user5','pwd':'davian','isDone':False},{'_id':'user6','pwd':'davian','isDone':False},{'_id':'user7','pwd':'davian','isDone':False},{'_id':'user8','pwd':'davian','isDone':False}])
collection_image = db.images
collection_log = db.log
collection_current = db.Current_toLabel
collection_before = db.Before_toLabel

total_image_list = sorted(os.listdir(os.path.join(APP_ROOT,CONST_IMAGE_PATH)))
total_num = len(total_image_list)

collection_image.insert([{"image_id" : total_image_list[i], "image_index" : i} for i in range(len(total_image_list))])

feature_list = []
key_list = []
attr_list2 = []

features = read_pck(CONST_PRETRAINED_FEATURE)[0]
attr_list = read_pck('attr_list.pickle')[0]

for each_key in sorted(attr_list):
    attr_list2.append(attr_list[each_key])

for each_key in sorted(features):
    feature_list.append(features[each_key])
    key_list.append(each_key)
feature_np = np.array(feature_list)

print("attr",attr_list2)

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
    cluster_data.append({"image_id":key_list[idx[0][np.argmax(dist)]], 
                         "image_list": [str(item) for item in list(idx[0])],
                         "image_id_list": [total_image_list[item] for item in list(idx[0])],
                         "x":str(center[0]), 
                         "y":str(center[1]), 
                         "count":str(len(list(idx[0])))
                         })

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
        return jsonify("good")
        
@app.route('/getAttribute', methods = ['GET','POST'])
def getAttribute():
    if request.method == "POST":
        json_received = request.form
        data = json_received.to_dict(flat=False)
        data_list = json.loads(data['jsonData'][0])
        print(data_list)
        print(attr_list[data_list['image_id']])
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

        db_image_list = [item['image_id'] for item in collection_image.find()]
        prelabeled_image_list = [item['image_id'] for item in collection_labeled.find({"user_id" : user_id, "adjective" : CONST_ADJECTIVE[keyword_index]})]        
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
            attribute_temp = copy.deepcopy(attr_list2)
           
            attribute_removed = removeFeature(attribute_temp, prelabeled_image_list).tolist()
 
            selectedAttribute = data['attribute'][0]

            collection_log.insert({"Time":time,"user_id": user_id, "What":"explore", "To":selectedAttribute})
            
            if data['label'][0] == "positive":

                appendImage(blue_list, possible_temp, attribute_temp, get_attribute_images(selectedAttribute,attribute_temp,0,CONST_BLUE_NUMBER))
                appendImage(red_list, possible_temp, attribute_temp, get_not_attribute_images(selectedAttribute,attribute_temp,0,CONST_RED_NUMBER))
                
                if len(possible_temp) >= CONST_NEUTRAL_NUMBER:
                    appendImage(neutral_list, possible_temp, attribute_temp, random.sample(range(len(possible_temp)),CONST_NEUTRAL_NUMBER))
                else:
                    appendImage(neutral_list, possible_temp, attribute_temp, random.sample(range(len(possible_temp)),len(possible_temp)))
            
            else:
                
                appendImage(blue_list, possible_temp, attribute_temp, get_not_attribute_images(selectedAttribute,attribute_temp,0,CONST_BLUE_NUMBER))
                appendImage(red_list, possible_temp, attribute_temp, get_attribute_images(selectedAttribute,attribute_temp,0,CONST_RED_NUMBER))
                
                if len(possible_temp) >= CONST_NEUTRAL_NUMBER:
                    appendImage(neutral_list, possible_temp, attribute_temp, random.sample(range(len(possible_temp)),CONST_NEUTRAL_NUMBER))
                else:
                    appendImage(neutral_list, possible_temp, attribute_temp, random.sample(range(len(possible_temp)),len(possible_temp)))
            


        current_todo = blue_list + neutral_list + red_list

        labeled = [total_image_list.index(item) for item in current_todo]        
        cluster = extractCluster(labeled,"image_id")
        print(cluster)

        for i in range(CONST_BATCH_NUMBER):
            if len(current_todo) > i:
                collection_current.update({"user_id":user_id , "index":i}, {"user_id":user_id , "index":i, "adjective": keyword_index, "image_id" : current_todo[i]})
            else:
                collection_current.update({"user_id":user_id , "index":i}, {"user_id":user_id , "index":i, "adjective": keyword_index, "image_id" : None})
        
        attribute_score = calculateAttribute(user_id, keyword_index)

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
    user_id = session.get("user_id")
    #data 추가하는 것 try except 문으로 또 걸어주기 (id, pwd)까지
    if request.method == "POST":
        blue_number = 0
        red_number = 0
        neutral_number = 0
        blue_list = []
        red_list = []
        neutral_list = []
        batch_list = []
        isNewset = None
        time = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

        json_received = request.form
        data = json_received.to_dict(flat=False)
        data_list = json.loads(data['jsonData'][0])
        # print(data_list[0])
        for item in data_list:
            item['user_id'] = user_id

        collection_log.insert({"Time":time,"user_id": user_id, "What":"confirm"})
        if data_list:
            collection_labeled.insert(data_list)

        keyword_index = collection_current.find({"user_id": user_id})[0]['adjective']
        
        imageStandard = choosingImage(data_list, CONST_ADJECTIVE[keyword_index])

        db_image_list = [item['image_id'] for item in collection_image.find()]
        prelabeled_image_list = [item['image_id'] for item in collection_labeled.find({"user_id" : user_id, "adjective" : CONST_ADJECTIVE[keyword_index]})]        
        possible_images = sorted(list(set(db_image_list) - set(prelabeled_image_list)))

        if not possible_images:
            if CONST_SAMPLING_MODE == "CLUSTER":
                blue_number = CONST_BLUE_NUMBER
                red_number = CONST_RED_NUMBER
                neutral_number = CONST_NEUTRAL_NUMBER
                isNewset = True
                keyword_index = keyword_index + 1
                   
                isnotFull = True
                total_cluster = copy.deepcopy(cluster_data)
                lastCluster = total_cluster[random.randint(0,len(total_cluster)-1)]['image_id']
                while(isnotFull):
                    for item in total_cluster:
                        if item['image_id'] == lastCluster:
                            if not item['image_id_list']:
                                print("empty!")
                                total_cluster.remove(item)
                                lastCluster = updateLastcluster(lastCluster,total_cluster,"index")

                            else:                        
                                print(len(item['image_id_list']))
                                batch_list.append(item['image_id_list'].pop())
                                if len(batch_list) == CONST_BATCH_NUMBER:
                                    print("false")
                                    isnotFull = False        
            else:
                blue_number = CONST_RANDOM_BLUE_NUMBER
                red_number = CONST_RANDOM_RED_NUMBER
                neutral_number = CONST_RANDOM_NEUTRAL_NUMBER
                isNewset = True
                keyword_index = keyword_index + 1
                db_image_list = [item['image_id'] for item in collection_image.find()]
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
            feature_temp = copy.deepcopy(feature_list)
            print("feature_temp", len(feature_temp))

            feature_removed = removeFeature(feature_temp, prelabeled_image_list)
            print("feature_removed", len(feature_removed))

            # print(imageStandard[0])
            # print(get_similar_images(imageStandard[0],feature_removed,6))
            # print([possible_temp[item] for item in get_similar_images(imageStandard[0],feature_removed,6)])

            # 여기서 모델로 사진 결정

            if CONST_SAMPLING_MODE == "RANDOM":
                # CONST_BLUE_NUMBER = 0
                # CONST_NEUTRAL_NUMBER = CONST_BATCH_NUMBER
                # CONST_RED_NUMBER = 0

                # if len(possible_temp) >= CONST_BLUE_NUMBER:
                #     appendImage(blue_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),CONST_BLUE_NUMBER))
                #     feature_removed = np.array(feature_temp)
                
                # else:
                #     appendImage(red_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),len(possible_temp)))
                #     feature_removed = np.array(feature_temp)
                # if len(possible_temp) >= CONST_RED_NUMBER:
                #     appendImage(red_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),CONST_RED_NUMBER))
                #     feature_removed = np.array(feature_temp)
                
                # else:
                #     appendImage(red_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),len(possible_temp)))
                #     feature_removed = np.array(feature_temp)
                if len(possible_temp) >= CONST_RANDOM_NEUTRAL_NUMBER:
                    appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),CONST_RANDOM_NEUTRAL_NUMBER))
                else:
                    appendImage(neutral_list, possible_temp, feature_temp, random.sample(range(len(possible_temp)),len(possible_temp)))
                blue_number = CONST_RANDOM_BLUE_NUMBER
                red_number = CONST_RANDOM_RED_NUMBER
                neutral_number = CONST_RANDOM_NEUTRAL_NUMBER

                batch_list = neutral_list

            elif CONST_SAMPLING_MODE == "CLUSTER":
                
                blue_number = CONST_BLUE_NUMBER
                red_number = CONST_RED_NUMBER
                neutral_number = CONST_NEUTRAL_NUMBER

                total_cluster = copy.deepcopy(cluster_data)
                print(len(total_cluster))
                for item in prelabeled_image_list:
                    for item2 in total_cluster:
                        if item in item2['image_id_list']:
                            item2['image_id_list'].pop(item2['image_id_list'].index(item))
                            if not item2['image_id_list']:
                                total_cluster.remove(item2)
                print(len(total_cluster))
                lastCluster = list(collection_labeled.find({"user_id" : user_id, "adjective" : CONST_ADJECTIVE[keyword_index]}).sort("_id",pymongo.DESCENDING).limit(1))[0]['cluster']
                print("current_cluster", lastCluster)
                
                count = 0
                for item in total_cluster:
                    if item['image_id'] != lastCluster:
                        count += 1
                if count == len(total_cluster):
                    lastCluster = updateLastcluster(data_list,total_cluster,"ajax")
                
                
                batch_list = []
                isnotFull = True
                while(isnotFull):
                    for item in total_cluster:
                        if item['image_id'] == lastCluster:
                            if not item['image_id_list']:
                                print("empty!")
                                total_cluster.remove(item)
                                if not total_cluster:
                                    isnotFull = False
                                    break
                                lastCluster = updateLastcluster(data_list,total_cluster,"ajax")

                            else:                        
                                print(len(item['image_id_list']))
                                batch_list.append(item['image_id_list'].pop())
                                if len(batch_list) == CONST_BATCH_NUMBER:
                                    print("false")
                                    isnotFull = False



        current_todo = batch_list

        labeled = [total_image_list.index(item['image_id']) for item in data_list]
        current_cluster_index = [total_image_list.index(item) for item in current_todo]
        current_cluster = extractCluster(current_cluster_index, "image_id")
        print(current_cluster)
        labeled_cluster = extractCluster(labeled,'image_id')
        label = np.array([d['label'] for d in data_list])

        for i in range(0,CONST_BATCH_NUMBER):
            if len(current_todo) > i:
                collection_current.update({"user_id":user_id , "index":i}, {"user_id":user_id , "index":i, "adjective": keyword_index, "image_id" : current_todo[i]})
            else:
                collection_current.update({"user_id":user_id , "index":i}, {"user_id":user_id , "index":i, "adjective": keyword_index, "image_id" : None})
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
            count = int(cluster_data[[item['image_id'] for item in cluster_data].index(outScore[i]["image_id"])]['count'])
            outScore[i]['score'] = outScore[i]['score'] / count

        attribute_score = calculateAttribute(user_id, keyword_index)
        print("blue", current_todo[0:CONST_BLUE_NUMBER])
        return jsonify({"blue":current_todo[0:blue_number], "neutral":current_todo[blue_number:blue_number+neutral_number], "red": current_todo[blue_number+neutral_number:blue_number+neutral_number+red_number],
                     "keyword": CONST_ADJECTIVE[keyword_index],
                    "image_count" : (int((total_num - len(possible_images))/CONST_BATCH_NUMBER)+1), 
                    "index": keyword_index,
                    "isNewset" : isNewset,
                    "score" : outScore,
                    "current_cluster" : current_cluster,
                    "positive_attr_list" : attribute_score[0],
                    "negative_attr_list" : attribute_score[1]
                                        })
        

@app.route('/index', methods = ['GET', 'POST'])
def index():

    blue_list = []
    red_list = []
    neutral_list = []
    
    print(session.get('logged_in'))
    if session.get("logged_in")==True:
        print(session.get("user_id"), session.get("logged_in"))
        
        user_id = session.get("user_id")
        db_image_list = [item['image_id'] for item in collection_image.find()]
        
        todo_images = [item for item in collection_current.find({"user_id" : user_id})]
        if todo_images:
            print("old")
            dictOfImg = { i : todo_images[i]['image_id'] for i in range(0,CONST_BATCH_NUMBER)}
            keyword_index = todo_images[-1]["adjective"]
            print("keyword",keyword_index)
        else:
            print("new")
            if CONST_SAMPLING_MODE == "RANDOM":
                batch_list = [total_image_list[item] for item in random.sample(range(len(total_image_list)),CONST_BATCH_NUMBER)]

            elif CONST_SAMPLING_MODE == "CLUSTER":
                batch_list = []
                isnotFull = True
                total_cluster = copy.deepcopy(cluster_data)
                lastCluster = total_cluster[random.randint(0,len(total_cluster)-1)]['image_id']
                while(isnotFull):
                    for item in total_cluster:
                        if item['image_id'] == lastCluster:
                            if not item['image_id_list']:
                                print("empty!")
                                total_cluster.remove(item)
                                lastCluster = updateLastcluster(lastCluster,total_cluster,"index")

                            else:                        
                                print(len(item['image_id_list']))
                                batch_list.append(item['image_id_list'].pop())
                                if len(batch_list) == CONST_BATCH_NUMBER:
                                    print("false")
                                    isnotFull = False        
            
            
            dictOfImg = { i : batch_list[i] for i in range(0,CONST_BATCH_NUMBER)}
            keyword_index = 0
            collection_current.insert([{'user_id' : user_id, "adjective" : 0, 'index' : i, 'image_id' : dictOfImg[i]} for i in range(0,CONST_BATCH_NUMBER)])

        outCluster = copy.deepcopy(cluster_data)

        current_cluster_index = []
        for item in dictOfImg.keys():
            if dictOfImg[item] != None:
                current_cluster_index.append(total_image_list.index(dictOfImg[item]))
        # current_cluster_index = [total_image_list.index(dictOfImg[item]) for item in dictOfImg.keys()]
        current_cluster = extractCluster(current_cluster_index,"image_id")

        labeled_data = list(collection_labeled.find({'user_id':user_id,'adjective':CONST_ADJECTIVE[keyword_index]},{'_id':0,'user_id':0,'adjective':0,'time':0}))
        
        labeled = [total_image_list.index(item['image_id']) for item in labeled_data]
        labeled_cluster = extractCluster(labeled,'index')

        for i in range(0,CONST_CLUSTER_NUMBER):
            sum_score = 0
            for j in range(len(labeled_cluster)):
                if labeled_cluster[j] == i:
                    sum_score += labeled_data[j]['label']
                    outCluster[i]['labeled'] = True
            outCluster[i]['score'] = str(sum_score/len(outCluster[i]['image_list']))

        count = int(len(labeled_data)/CONST_BATCH_NUMBER)+1
        
        # 여기서 첫 세트 사진 결정
        # 형용사 결정
        # user_id = str(user_id)
        images = json.dumps(dictOfImg)
        label = json.dumps(labeled_data)
        current_cluster_json = json.dumps(current_cluster)
        cluster_json = json.dumps(outCluster)

        attribute_score = calculateAttribute(user_id, keyword_index)

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


