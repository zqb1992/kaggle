#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@authon:
@file:Titanic_XGBoost.py
@time:2018/11/28
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
import time
startTime = time.time()

#加载数据
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

#清洗数据
def clean_data(titanic):#填充空数据 和 把string数据转成integer表示
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    # child
    titanic["child"] = titanic["Age"].apply(lambda x: 1 if x < 15 else 0)

    # sex
    titanic["sex"] = titanic["Sex"].apply(lambda x: 1 if x == "male" else 0)

    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    # embark
    def getEmbark(Embarked):
        if Embarked == "S":
            return 1
        elif Embarked == "C":
            return 2
        else:
            return 3
    titanic["embark"] = titanic["Embarked"].apply(getEmbark)

    # familysize
    titanic["fimalysize"] = titanic["SibSp"] + titanic["Parch"] + 1

    # cabin
    def getCabin(cabin):
        if cabin == "N":
            return 0
        else:
            return 1
    titanic["cabin"] = titanic["Cabin"].apply(getCabin)

    # name
    def getName(name):
        if "Mr" in str(name):
            return 1
        elif "Mrs" in str(name):
            return 2
        else:
            return 0
    titanic["name"] = titanic["Name"].apply(getName)

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic

train_data = clean_data(train)
test_data = clean_data(test)

features = ["Pclass", "sex", "child", "fimalysize", "Fare", "embark", "cabin"]

# 简单初始化xgb的分类器就可以
clf =XGBClassifier(n_estimators=166,learning_rate=0.11, max_depth=2,
silent=True, objective='binary:logistic')

# 设置boosting迭代计算次数
param_test = {
    'n_estimators':range(100,200,2),
    'max_depth': range(2, 7, 1),
    'min_samples_leaf':range(2,10,1),
    'learning_rate':np.arange(0.01,0.2,0.01)
}
grid_search = GridSearchCV(estimator = clf, param_grid = param_test,
scoring='accuracy', cv=5)
grid_search.fit(train_data[features], train_data["Survived"])
print(grid_search.grid_scores_)
print(grid_search.best_params_)
print(grid_search.best_score_)

predictions = grid_search.predict(test_data[features])
PassengerId = test['PassengerId']
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
StackingSubmission.to_csv("../out/Stack.csv", index=False)

endTime=time.time()
print(u"耗时：%f s" %(endTime-startTime))