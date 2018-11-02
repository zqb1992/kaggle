#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@authon:
@file:.py
@time:2018/11/02
"""
#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@authon:
@file:.py
@time:2018/11/01
"""
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np

# from lightgbm import LGBMClassifier
startTime = time.time()
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


clf = DecisionTreeClassifier(random_state=0)
#参数调优，找最佳参数
param_test1 = {'max_depth':np.arange(1,11,1)}
gsearch1 = GridSearchCV(estimator = clf, param_grid = param_test1, scoring='accuracy',iid=False,cv=5)
gsearch1.fit(Xtrain,xlabel)
print( gsearch1.best_params_, gsearch1.best_score_)


clf.fit(Xtrain,xlabel)
print(clf.n_classes_)

#输出正确率
print("the right rate is:",clf.score(Ytrain,ylabel))

Ytrain_pre = clf.predict(Ytrain)
print(classification_report(ylabel,Ytrain_pre))


#结果预测
result=clf.predict(data_pca[len(train_data):])
with open("../out/predict_dt.csv", 'w') as fw:
    with open('../data/sample_submission.csv') as pred_file:
        fw.write('{},{}\n'.format('ImageId', 'Label'))
        for i,line in enumerate(pred_file.readlines()[1:]):
            splits = line.strip().split(',')
            fw.write('{},{}\n'.format(splits[0],result[i]))

endTime = time.time()
print(u"耗时：%f s" %(endTime-startTime))