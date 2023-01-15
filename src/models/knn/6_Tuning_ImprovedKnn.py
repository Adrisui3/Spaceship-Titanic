#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: marcosesquivelgonzalez

Se buscan los valores de k y p óptimos para el ImprovedKnn haciendo como siempre SCV con 10 folds.

Los datos se encuentran imputados por knn y mergeadas las columnas numéricas.

Además, este script se corrió varias veces al darme cuenta que la imputación por knn estaba usando k=30, 
lo cual es mal número al ser par. Por tanto, se volvió a correr haciendo la imputación por knn con k=43.

Por otro lado, se probó a editar el ImprovedKnn usando la mediana en lugar de la media en la fórmula de
votación así como con otros valores de k para la ponderación de características. Esto se hacía editando
antes el archivo 'ImprovedKnn.py'.

"""
import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import ImprovedKnn
import pandas as pd

train_raw = utils.load_train_KnnImp()
train_raw = utils.merge_numerical(train_raw) 
train_y = train_raw.Transported
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
colnames_train = train_X.columns

test_raw = utils.load_test_KnnImp()
test_raw = utils.merge_numerical(test_raw)
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))
colnames_tst = test.columns

#RobScal = RobustScaler()
RobScal = Normalizer()
train_scaled_array = RobScal.fit_transform(train_X)
train_scaled = pd.DataFrame(train_scaled_array,columns=colnames_train)

test_scaled_array = RobScal.transform(test)
test_scaled = pd.DataFrame(test_scaled_array,columns=colnames_tst)



skf = StratifiedKFold(n_splits = 10, shuffle=True,random_state = 1)
print('Improved knn con K=3,5,7\n')
best_score = 0
for k in range(20,84):
    cv = {"test_score":[]}
    for train_idx, test_idx in skf.split(X = train_scaled, y = train_y):
        
        Xi_train, yi_train = train_scaled.loc[train_idx], train_y.loc[train_idx]
        Xi_test, yi_test = train_scaled.loc[test_idx], train_y.loc[test_idx]
     
        test_preds = ImprovedKnn.improved_knn(n_neighbors=k, train=Xi_train, train_classes=yi_train, test=Xi_test)
        acc_test = accuracy_score(yi_test, test_preds)
        cv["test_score"].append(acc_test)
        
    score = np.mean(cv['test_score'])
    print('%.4f \u00B1 %.4f %.d' %(score,np.std(cv['test_score']),k))
    print('%.4f \u00B1 %.4f %.d' %(np.mean(cv['train_score']),np.std(cv['train_score']),k),'\n')
    if score>best_score:
        best_score = score
        k_optim = k

print('El mejor resultado es para k=',k_optim,':',best_score)
    
