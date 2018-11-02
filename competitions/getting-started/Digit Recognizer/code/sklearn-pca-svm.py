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
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

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

# #PCA维数分析
# pca=PCA()
# pca.fit(train_data,train_label)
# ratio=pca.explained_variance_ratio_
# print("pca.components_",pca.components_.shape)
# print("pca_var_ratio",pca.explained_variance_ratio_.shape)
# plt.plot([i for i in range(train_data.shape[1]-600)],
#          [np.sum(ratio[:i+1]) for i in range(train_data.shape[1]-600)])
#
# plt.xticks(np.arange(train_data.shape[1]-600,step=10))
# plt.yticks(np.arange(0,1.01,0.05))
# plt.title(u"PCA维数分析")
# plt.xlabel(u"维数")
# plt.ylabel(u"方差")
# plt.grid()
# plt.savefig("../image/pca_analyse.png")
# plt.show()

COMPONENT_NUM = 80
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train_data)

print(sum(pca.explained_variance_ratio_))
train_data = pca.transform(train_data)

#将降维后的训练集分为训练集和验证集两部分
train_data,dev_data,train_label,dev_label = train_test_split(train_data,train_label,test_size=0.1,random_state=1)

print(train_data.shape)
print(dev_data.shape)



# print('SVM Params...')
# #参数调优
# grid = GridSearchCV(SVC(),param_grid={"C":[0.1,1,10],"gamma":[1,0.1,0.01]},cv=4)
# grid.fit(train_data,train_label)
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))



#svm
print('Train SVM...')

#C较大，gamma较大时，会有更多的支持向量，模型会比较复杂，容易过拟合
svc = SVC(C=6, kernel='rbf')
svc.fit(train_data, train_label)

#交叉验证
print(svc.score(train_data,train_label))
print(svc.score(dev_data,dev_label))

# pre_dev_label = svc.predict(dev_data)
# zeroLabel = pre_dev_label-dev_label
# rightCount = np.sum(zeroLabel == 0)
# print('the right rate is:',float(rightCount)/len(zeroLabel))

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
predict = svc.predict(test_data)

#保存测试结果
print('Saving...')
with open('../out/predict_svm.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')

endTime=time.time()
print(u"耗时：%f s" %(endTime-startTime))