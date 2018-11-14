#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@authon:
@file:sklearn_boosting.py
@time:2018/11/12
"""
print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import datetime

#load data
print('Load data')
train_data=pd.read_csv(r'../data/train.csv')
test_data=pd.read_csv(r'../data/test.csv')

#按行连接起来
data=pd.concat([train_data,test_data],axis=0).reset_index(drop=True)
#删除data中的label列
data.drop(['label'],axis=1,inplace=True)
label=train_data.label

#PCA处理
print('PCA')
pca=PCA(n_components=35, random_state=1)
data_pca=pca.fit_transform(data)
#定义交叉验证
Xtrain,Ytrain,xlabel,ylabel=train_test_split(data_pca[0:len(train_data)],label,test_size=0.1, random_state=34)

print('Boosting')
starttime = datetime.datetime.now()
adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=500)

#adaboost算法本身没有限制使用神经网络作为基学习器。但是sklearn的adaboost算法对基学习器有要求，
# 必须支持样本权重和类属性 classes_ 以及n_classes_
#MLPClassifier没有权重，也没有 n_classes_ ，因此不能使用。
#adb = AdaBoostClassifier(MLPClassifier(hidden_layer_sizes=(20,)),n_estimators=100)

adb.fit(Xtrain,xlabel)
print(adb.score(Ytrain,ylabel))

result = adb.predict(data_pca[len(train_data):])

print('Saving...')
with open('../out/sklearn_adaboosting.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in result:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')

endtime = datetime.datetime.now()
print((endtime - starttime).seconds)