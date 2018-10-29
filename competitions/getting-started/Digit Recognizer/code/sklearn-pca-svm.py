#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
@authon: ZQB
@file:sklearn-pca-svm.py
@time:2018/10/25
"""
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC

startTime = time.time()

COMPONENT_NUM = 35

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
print('Reduction...')
train_label = np.array(train_label)
train_data = np.array(train_data)
# pca = PCA()
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train_data)

print(sum(pca.explained_variance_ratio_))

train_data = pca.transform(train_data)

#将降维后的训练集分为训练集和验证集两部分
train_data,dev_data,train_label,dev_label = train_test_split(train_data,train_label,test_size=0.1,random_state=1)

print(train_data.shape)
print(dev_data.shape)


#svm
print('Train SVM...')
svc = SVC(C=4, kernel='rbf')
svc.fit(train_data, train_label)


#交叉验证
pre_dev_label = svc.predict(dev_data)
zeroLabel = pre_dev_label-dev_label
rightCount = np.sum(zeroLabel == 0)
print('the right rate is:',float(rightCount)/len(zeroLabel))

endTime=time.time()
print(u"耗时：%f s" %(endTime-startTime))

print('Read testing data...')
with open('../data/test.csv', 'r') as reader:
    reader.readline()
    test_data = []
    for line in reader.readlines():
        pixels = list(map(int, line.rstrip().split(',')))
        test_data.append(pixels)
print('Loaded ' + str(len(test_data)))

print('Predicting...')
test_data = np.array(test_data)
test_data = pca.transform(test_data)
predict = svc.predict(test_data)

print('Saving...')
with open('../out/digit_recognizer.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')

endTime=time.time()
print(u"耗时：%f s" %(endTime-startTime))