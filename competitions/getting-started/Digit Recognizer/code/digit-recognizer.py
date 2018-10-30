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
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


#加载数据
def load_data():
    dataTrain = pd.read_csv(r"../data/train.csv")
    dataTest = pd.read_csv(r"../data/test.csv")

    train_data = dataTrain.values[:,1:]
    train_label = dataTrain.values[:,0]
    test_data = dataTest.values[:,:]

    return train_data,train_label,test_data

train_data,train_label,test_data = load_data()

#pca processing
print('PCA...')
train_label = np.array(train_label)
train_data = np.array(train_data)


#求每一列的均值
meanVal = np.mean(train_data,axis=0)
print(meanVal.shape)
#归一化
meanRemoved = train_data-meanVal;
print(meanRemoved.shape)
#求协方差矩阵
covData = np.cov(meanRemoved,rowvar=0)
#计算特征值和特征向量
eig_vals,eig_vecs = np.linalg.eig(covData)
#从大到小排序
eig_vals_id = np.argsort(-eig_vals)

print(eig_vals_id.shape)
print(eig_vals_id[1:10]);

#特征值对方差的影响
total = float(np.sum(eig_vals))   #总特征值和
var_exp = [float(eig_vals[eig_vals_id[i]])/total for i in range(0,len(eig_vals_id))]
cum_var_exp = np.cumsum(var_exp)

# for i in range(0, len(eig_vals_id)):
#     print('主成分：%s, 方差占比：%s%%, 累积方差占比：%s%%' % (
#     format(i + 1, '2.0f'), format(var_exp[i] * 100, '4.2f'),
#     format(cum_var_exp[i] * 100, '4.1f')))

plt.plot([i for i in range(train_data.shape[1]-684)],cum_var_exp[0:100])
plt.xticks(np.arange(train_data.shape[1]-684,step=10))
plt.yticks(np.arange(0,1.01,0.05))
plt.title(u"PCA维数分析")
plt.xlabel(u"维数")
plt.ylabel(u"方差")
plt.grid()
plt.show()




















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

















