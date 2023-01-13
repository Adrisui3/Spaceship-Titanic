# PATH for utils functions
import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
ROOT = os.path.abspath(os.path.join(SRC, os.pardir))
sys.path.append(SRC)

#Libraries 
import utils 
import numpy as np
import pandas as pd
from mixed_naive_bayes import MixedNB 
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import cross_validate

print('--- LOAD DATA ---')
train_raw = pd.read_csv(ROOT + '/data/train_nooutliers.csv')
train_X = utils.one_hot_encode(train_raw.drop(['Transported', 'PassengerId'], 
                                              axis = 1))
train_y = train_raw['Transported']

num_features = train_raw.select_dtypes(exclude=['object', 'bool']).columns
cat_features = train_X.drop(num_features, axis = 1).columns

test_raw = utils.load_test()
test = utils.one_hot_encode(test_raw.drop(['PassengerId'], axis = 1))

print('--- NORMALIZER ---')
train_Xnorm = train_X.copy()
train_Xnorm.loc[:, num_features] = Normalizer().fit_transform(train_Xnorm.loc[:, num_features])

test_norm = test.copy()
test_norm.loc[:, num_features] = Normalizer().fit_transform(test_norm.loc[:, num_features])



