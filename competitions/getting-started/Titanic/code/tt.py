#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@authon:
@file:.py
@time:2018/01/19
"""
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print train.info()
print test.info()

selected_features = ['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']
