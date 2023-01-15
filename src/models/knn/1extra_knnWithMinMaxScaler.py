#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: marcosesquivelgonzalez

MinMaxScaler da malos resultados al haber outliers en algunos de los atributos numéricos. Estos resultados
se añaden al mismo .txt que para 1_knn_tuningK_MetricScalerWeights.py al hacerse un análisis equivalente
también para los datos con imputación por moda y sin mergear numéricas.

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

train_raw = utils.load_train()
#train_raw = utils.merge_numerical(train_raw) #PARA MERGEAR LAS COLUMNAS VISTAS EN EDA

train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
colnames = train_X.columns
train_y = train_raw.Transported

MMaxScal = MinMaxScaler()
df_array = MMaxScal.fit_transform(train_X)
train_X_scaled = pd.DataFrame(df_array,columns=colnames)



weights = ["uniform", "distance"]

print('Para MinMaxScaler data')

for w in weights:
    print('Pesos:',w,'\n')
    print('CV score, k_optim, p')
    for p_ in [1,2]:
        optim_score=0
        for i in range(5,50):
            
    
                
            score_cv = cross_validate(estimator = KNeighborsClassifier(n_neighbors=i, p = p_,weights=w),
                                          X = train_X_scaled, y = train_y, cv = 10, n_jobs = -1)
            #print(round(np.mean(score_cv["test_score"]),5))
            score_test = np.mean(score_cv["test_score"])
            if score_test > optim_score:
                optim_score = score_test
                k_optim = i
            
        print(optim_score, k_optim ,p_)