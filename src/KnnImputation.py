#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 00:27:16 2023

@author: marcosesquivelgonzalez
"""
import os
import pandas as pd
from sklearn.impute import KNNImputer


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

train_raw  = pd.read_csv(ROOT + '/data/train_pr_to_KnnImp.csv')
test_raw = pd.read_csv(ROOT + '/data/test_pr_to_KnnImp.csv')

imputer = KNNImputer(n_neighbors=30)

imputer.fit(train_raw)

train_raw_knnI = imputer.transform(train_raw)
test_raw_knn =imputer.transform(test_raw)