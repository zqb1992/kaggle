#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@authon:ZQB
@file:sklearn_RF.py
@time:2018/11/01
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# from lightgbm import LGBMClassifier

train_data=pd.read_csv(r"../data/train.csv")
test_data=pd.read_csv(r"../data/test.csv")

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

clf=RandomForestClassifier(n_estimators=110,max_depth=5,min_samples_split=2, min_samples_leaf=1,random_state=34,oob_score=True)

# clf=LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)

# param_test1 = {'n_estimators':np.arange(10,150,10),'max_depth':np.arange(1,11,1)}
# gsearch1 = GridSearchCV(estimator = clf, param_grid = param_test1, scoring='accuracy',iid=False,cv=5)
# gsearch1.fit(Xtrain,xlabel)
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

clf.fit(Xtrain,xlabel)

#使用袋外错误率估计正确率
print(clf.oob_score_)
print(clf.n_classes_)

#输出正确率
print("the right rate is:",clf.score(Ytrain,ylabel))


result=clf.predict(data_pca[len(train_data):])

with open("../out/predict_rf.csv", 'w') as fw:
    with open('../data/sample_submission.csv') as pred_file:
        fw.write('{},{}\n'.format('ImageId', 'Label'))
        for i,line in enumerate(pred_file.readlines()[1:]):
            splits = line.strip().split(',')
            fw.write('{},{}\n'.format(splits[0],result[i]))