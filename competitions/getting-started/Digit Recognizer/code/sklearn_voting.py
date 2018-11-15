#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@authon:ZQB
@file:sklearn_voting.py
@time:2018/11/15
"""
import numpy as np
import pandas as pd
import datetime

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

import matplotlib.pyplot as plt

starttime=datetime.datetime.now()
#load data
train_data=pd.read_csv(r'../data/train.csv')
test_data=pd.read_csv(r'../data/test.csv')
#按行连接起来
data=pd.concat([train_data,test_data],axis=0).reset_index(drop=True)
#删除data中的label列
data.drop(['label'],axis=1,inplace=True)
label=train_data.label

#PCA处理
pca=PCA(n_components=35, random_state=1)
data_pca=pca.fit_transform(data)
#定义交叉验证
Xtrain,Ytrain,xlabel,ylabel=train_test_split(data_pca[0:len(train_data)],label,test_size=0.1, random_state=1)

svc = SVC(C=6, kernel='rbf')
svc.fit(Xtrain,xlabel)
print("the SVM's right rate is:",svc.score(Ytrain,ylabel))

dt = DecisionTreeClassifier(random_state=0)
dt.fit(Xtrain,xlabel)
print("the DT's right rate is:",dt.score(Ytrain,ylabel))

knn = KNeighborsClassifier(n_neighbors =10,leaf_size=40)
knn.fit(Xtrain,xlabel)
print("the KNN's right rate is:",knn.score(Ytrain,ylabel))

nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, alpha=1e-1,
                    solver='adam', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=0.001)
nn.fit(Xtrain,xlabel)
print("the NN's right rate is:",nn.score(Ytrain,ylabel))

rf=RandomForestClassifier(n_estimators=110,max_depth=5,min_samples_split=2, min_samples_leaf=1,random_state=34,oob_score=True)
rf.fit(Xtrain,xlabel)
print("the RF's right rate is:",rf.score(Ytrain,ylabel))

adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=500)
adb.fit(Xtrain,xlabel)
print("the Adb's right rate is:",adb.score(Ytrain,ylabel))

gbm = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=11, min_samples_leaf =100,
               min_samples_split =1200, max_features='sqrt', subsample=0.8, random_state=10)
gbm.fit(Xtrain,xlabel)
print("the GBDT's right rate is:",gbm.score(Ytrain,ylabel))


clf = VotingClassifier(estimators=[('dt',dt),('knn',knn),('svc',svc),('nn',nn),('rf',rf),('adb',adb),('gbm',gbm)],voting='hard')
clf.fit(Xtrain,xlabel)

endtime=datetime.datetime.now()
print("耗时%f:"%(endtime-starttime).seconds)
print("the right rate is:",clf.score(Ytrain,ylabel))

#结果预测
result=clf.predict(data_pca[len(train_data):])
with open("../out/predict_voting.csv", 'w') as fw:
    with open('../data/sample_submission.csv') as pred_file:
        fw.write('{},{}\n'.format('ImageId', 'Label'))
        for i,line in enumerate(pred_file.readlines()[1:]):
            splits = line.strip().split(',')
            fw.write('{},{}\n'.format(splits[0],result[i]))
