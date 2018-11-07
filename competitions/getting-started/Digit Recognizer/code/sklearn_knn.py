#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@authon:
@file:sklearn_knn.py
@time:2018/11/06
"""

import time
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,roc_curve,auc,roc_auc_score
from sklearn.preprocessing import label_binarize
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

COMPONENT_NUM = 45
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train_data)

print(sum(pca.explained_variance_ratio_))
train_data = pca.transform(train_data)

#将降维后的训练集分为训练集和验证集两部分
train_data,dev_data,train_label,dev_label = train_test_split(train_data,train_label,test_size=0.1,random_state=1)

#为了绘制多分类下的ROC曲线
n_class=10
dev_one_hot = label_binarize(dev_label, np.arange(n_class))  #装换成类似二进制的编码


# # print('KNN Params...')
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


#knn
print('Train KNN...')
knn = KNeighborsClassifier(n_neighbors =10,leaf_size=40)
knn.fit(train_data,train_label)


#交叉验证
print(knn.score(train_data,train_label))
# print(knn.score(dev_data,dev_label))

# #查看召回率，准确率和f1分值
# dev_pre = knn.predict(dev_data)
# print(classification_report(dev_label,dev_pre))

dev_pre_proba = knn.predict_proba(dev_data)
print(dev_pre_proba.shape)
print('调用函数roc_auc_score：', roc_auc_score(dev_one_hot, dev_pre_proba, average='micro'))
#绘制ROC曲线
false_positive_rate,true_positive_rate,thresholds=roc_curve(dev_one_hot.ravel(), dev_pre_proba.ravel())
roc_auc=auc(false_positive_rate, true_positive_rate)
print('调用函数auc：', roc_auc)

#绘制整体模型的ROC曲线
plt.figure()
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()


# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc1 = dict()
for i in range(n_class):
    fpr[i], tpr[i], _ = roc_curve(dev_one_hot[:, i], dev_pre_proba[:, i])
    roc_auc1[i] = auc(fpr[i], tpr[i])

#绘制每一类的ROC曲线
plt.figure()
for i in range(n_class):
    plt.plot(fpr[i], tpr[i],  lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc1[i]))

plt.title('ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()


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
predict = knn.predict(test_data)

#保存测试结果
print('Saving...')
with open('../out/predict_knn.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')

endTime=time.time()
print(u"耗时：%f s" %(endTime-startTime))
