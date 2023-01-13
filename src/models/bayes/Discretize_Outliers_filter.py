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
train_X = pd.read_csv(ROOT + '/data/train_discretize_oh.csv')
train_y = utils.load_train()['Transported']

test_raw = pd.read_csv(ROOT + '/data/test_discretize_oh.csv')

print('--- CROSS VALIDATION --- ')
cv = cross_validate(
    estimator=MixedNB(categorical_features='all'),
    X = train_X, y = train_y, 
    return_train_score=True, 
    cv = 10, 
    n_jobs = -1, 
    verbose = 1
)
print('Result of the cv in test: ', np.mean(cv['test_score']))
print('Result of the cv in train: ', np.mean(cv['train_score']))