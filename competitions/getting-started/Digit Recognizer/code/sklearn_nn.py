#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@authon:
@file:sklearn_nn.py
@time:2018/11/07
"""
print(__doc__)

import time
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,roc_curve,auc
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

startTime = time.time()

#load train data
print('Read training data...')
with open('../data/train.csv', 'r') as reader:
    reader.readline()
    train_label = []
    train_data = []
    for line in reader.readlines():
        data = list(map(int, line.rstrip().split(',')))
        train_label.append(data[0])
        train_data.append(data[1:])
print('Loaded ' + str(len(train_label)))


#pca processing
print('PCA...')
train_label = np.array(train_label)
train_data = np.array(train_data)

COMPONENT_NUM = 30
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train_data)

print(sum(pca.explained_variance_ratio_))
train_data = pca.transform(train_data)

#将降维后的训练集分为训练集和验证集两部分
train_data,dev_data,train_label,dev_label = train_test_split(train_data,train_label,test_size=0.1,random_state=1)

# # print('NN Params...')
# #网格搜索进行参数调优
# k_range = list(range(4,11))
# leaf_range = list(range(30,40))
# weight_options = ['uniform','distance']
#
# param_gridknn = dict(n_neighbors = k_range,weights = weight_options,leaf_size=leaf_range)
#
# grid = GridSearchCV(KNeighborsClassifier(),param_gridknn,cv=10)
# grid.fit(train_data,train_label)
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))
#
# endTime=time.time()
# print(u"耗时：%f s" %(endTime-startTime))


#nn
print('Train NN...')
# nn = MLPClassifier(hidden_layer_sizes=(100,20), activation='relu', alpha=0.0001,learning_rate='constant',
#                    learning_rate_init=0.001,max_iter=200, shuffle=True, random_state=1)
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, alpha=1e-1,
                    solver='adam', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=0.001)
nn.fit(train_data,train_label)


#交叉验证
print(nn.score(train_data,train_label))
print(nn.score(dev_data,dev_label))

#查看召回率，准确率和f1分值
dev_pre = nn.predict(dev_data)
print(classification_report(dev_label,dev_pre))


#以下是预测部分
#读测试数据
print('Read testing data...')
with open('../data/test.csv', 'r') as reader:
    reader.readline()
    test_data = []
    for line in reader.readlines():
        pixels = list(map(int, line.rstrip().split(',')))
        test_data.append(pixels)
print('Loaded ' + str(len(test_data)))

#预测
print('Predicting...')
test_data = np.array(test_data)
test_data = pca.transform(test_data)
predict = nn.predict(test_data)

#保存测试结果
print('Saving...')
with open('../out/predict_nn.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')

endTime=time.time()
print(u"耗时：%f s" %(endTime-startTime))



