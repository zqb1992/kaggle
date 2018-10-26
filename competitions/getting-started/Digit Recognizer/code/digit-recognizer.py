#!/usr/bin/python 
# coding: utf-8

"""
@Time : 2018/10/25 21:56 
@Author : ZQB
@File : digit-recognizer.py 
@Dec : defining some basic functions
@Github : https://github.com/zqb1992/kaggle
"""

import pandas as pd
import numpy as np
import csv
from sklearn.decomposition import PCA

COMPONENT_NUM = 40

#加载数据
def load_data():
    dataTrain = pd.read_csv(r"../data/train.csv")
    dataTest = pd.read_csv(r"../data/test.csv")

    train_data = dataTrain.values[:,1:]
    train_label = dataTrain.values[:,0]
    test_data = dataTest.values[:,:]

    return train_data,train_label,test_data

load_data()

#pca处理
def DATA_PCA(train_data,test_data,component_num):
    trainData = np.array(train_data)
    testData = np.array(test_data)
    pca = PCA(n_components=component_num,whiten=TRUE)
    pca.fit(trainData)
    #输出所选择维数所占的方差比
    print(sum(pca.explained_variance_ratio_))
    #转化训练集
    train_data = pca.transform(trainData)
    test_data = pca.transform((testDataata))
    return train_data,test_data

#模型训练
def svm_model(train_data,train_label):
    print('Train SVM...')
    svc = SVC(C=4, kernel='rbf')
    svc.fit(train_data, train_label)
    return svc

#保存数据
def SaveResult(csvName,result):
    with open(csvName, 'w') as writer:
        writer.write('"ImageId","Label"\n')
        count = 0
        for res in result:
            count += 1
            writer.write(str(count) + ',"' + str(res) + '"\n')

















