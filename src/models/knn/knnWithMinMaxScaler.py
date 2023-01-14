#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:04:39 2023

@author: marcosesquivelgonzalez

MinMaxScaler da malos resultados al tener outliers presentes en algunos de los atributos numéricos
"""
import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate

train_raw = utils.load_train_KnnImp()
#train_raw = utils.load_train()
train_raw = utils.merge_numerical(train_raw) #PARA MERGEAR LAS COLUMNAS VISTAS EN EDA

train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
colnames = train_X.columns
train_y = train_raw.Transported

MMaxScal = MinMaxScaler()
df_array = MMaxScal.fit_transform(train_X)
train_X_scaled = pd.DataFrame(df_array,columns=colnames)



def my_weights(dist):
    return np.ones(18).reshape(dist.shape)

knn = KNeighborsClassifier(weights=my_weights)

knn.fit(train_X_scaled,train_y)

print(np.mean(knn.predict(train_X_scaled)==train_y))









for p_ in [1,2]:
    optim_score=0
    for i in range(5,50):
        

            
        score_cv = cross_validate(estimator = KNeighborsClassifier(n_neighbors=i, p = p_),
                                      X = train_X_scaled, y = train_y, cv = 10, n_jobs = -1)
        #print(round(np.mean(score_cv["test_score"]),5))
        score_test = np.mean(score_cv["test_score"])
        if score_test > optim_score:
            optim_score = score_test
            k_optim = i
            
    print(optim_score, k_optim ,p_)