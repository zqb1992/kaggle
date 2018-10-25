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

def load_data():
    dataTrain = pd.read_csv(r"../data/train.csv")
    dataTest = pd.read_csv(r"../data/test.csv")

    train_data = dataTrain.values[:,1:]
    train_label = dataTrain.values[:,0]
    test_data = dataTest.values[:,:]

    return train_data,train_label,test_data




load_data()
