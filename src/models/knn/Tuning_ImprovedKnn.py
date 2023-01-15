#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 21:42:29 2023

@author: marcosesquivelgonzalez
"""
import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
from sklearn.preprocessing import RobustScaler
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

RobScal = RobustScaler()

train_scaled_array = RobScal.fit_transform(train_X)
train_scaled = pd.DataFrame(train_scaled_array,columns=train_X.columns)

test_scaled_array = RobScal.transform(test)
test_scaled = pd.DataFrame(test_scaled_array,columns=test.columns)



skf = StratifiedKFold(n_splits = 10, shuffle=True,random_state = 1)

print('Improved knn con K=3,5,7\n')
best_score = 0
for k in range(10,70):
 
    cv = {"test_score":[]}
    for train_idx, test_idx in skf.split(X = train_scaled, y = train_y):
        
        Xi_train, yi_train = train_scaled.loc[train_idx], train_y.loc[train_idx]
        Xi_test, yi_test = train_scaled.loc[test_idx], train_y.loc[test_idx]
     
        test_preds = ImprovedKnn.improved_knn(n_neighbors=k, train=Xi_train, train_classes=yi_train, test=Xi_test)
        acc_test = accuracy_score(yi_test, test_preds)
        
        cv["test_score"].append(acc_test)
    score = np.mean(cv['test_score'])
    print('%.4f \u00B1 %.4f %.d' %(score,np.std(cv['test_score']),k))
    
    if score>best_score:
        best_score = score
        k_optim = k

print('El mejor resultado es para k=',k_optim,':',best_score)
    
