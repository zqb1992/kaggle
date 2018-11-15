#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@authon:ZQB
@file:XGBoost.py
@time:2018/11/14
"""
import numpy as np
import pandas as pd
import datetime

import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

starttime = datetime.datetime.now()
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
Xtrain,Ytrain,xlabel,ylabel=train_test_split(data_pca[0:len(train_data)],label,test_size=0.1, random_state=34)

params={
'booster':'gbtree',
# 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
'objective': 'multi:softmax', 
'num_class':10, # 类数，与 multisoftmax 并用
'gamma':0.05,  # 用于控制是否后剪枝的参数 在树的叶子节点下一个分区的最小损失，越大算法模型越保守。[0:]
'max_depth':3, # 生成树的深度，典型值[3:10]

#'lambda':450,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#'max_leaf_nodes':100, #叶子节点最大数量
'subsample':0.5, # 随机采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
'colsample_bytree':0.7, # 生成树时进行的列采样比率 (0:1]，与GBM中的max_feature参数类似

#'min_child_weight':12, # 最小叶子节点样本权重和，这个参数默认是 1，是每个叶子里面h的和至少是多少，对正负样本不均衡时的 0-1分类而言假设h在0.01附近，min_child_weight为1意味着叶子节点中最少需要包含100个样本。这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易过拟合。 

'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.1, # 如同学习率,典型值[0.01-0.2]
'seed':10, #随机数种子
'nthread':2,# cpu 线程数,根据自己U的个数适当调整
}
#转换成list
plst = list(params.items())

#将数据转化成xgboost矩阵
xgtrain=xgb.DMatrix(Xtrain,xlabel)
xgdev=xgb.DMatrix(Ytrain,ylabel)
xgtest=xgb.DMatrix(data_pca[len(train_data):])

num_rounds=500
watchlist = [(xgtrain, 'train'),(xgdev, 'val')]
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
model.save_model('./model/xgb.model') # 用于存储训练出的模型
print("best best_ntree_limit",model.best_ntree_limit) 

#交叉验证
pre_ytrain = model.predict(xgdev)
zeroLabel = pre_ytrain-ylabel
rightCount = np.sum(zeroLabel == 0)
print('the right rate is:',float(rightCount)/len(zeroLabel))


preds = model.predict(xgtest,ntree_limit=model.best_iteration)
# 将预测结果写入文件，方式有很多，自己顺手能实现即可
np.savetxt('xgboost_MultiSoftmax.csv',np.c_[range(1,len(test_data)+1),preds],
                delimiter=',',header='ImageId,Label',comments='',fmt='%d')

endtime = datetime.datetime.now()
print(u"耗时：%f s" %(endTime-startTime).seconds)