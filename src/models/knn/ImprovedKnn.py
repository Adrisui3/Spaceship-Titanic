#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 19:47:31 2023

@author: marcosesquivelgonzalez

Se intenta implementar un nuevo knn basado en el paper:
    An Improved kNN Based on Class Contribution and Feature Weighting
    -HUANG Jie1,2, WEI Yongqing3,*,YI Jing2,4 and LIU Mengdi1,
"""

import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import KNeighborsMixin
import numpy as np
import pandas as pd

train_raw = utils.load_train_KnnImp()
train_raw = utils.merge_numerical(train_raw) 
    
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))

RobScal = RobustScaler()
df_array = RobScal.fit_transform(train_X)
train_X_RobScal = pd.DataFrame(df_array,columns=train_X.columns)
    
    
def improved_knn(n_neighbors=30,weights,train_data,test_data):
    
    
    
    
    
    
    
    
    
    
    
    
    