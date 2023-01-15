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
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import cross_validate

print('--- LOAD DATA ---')
train = pd.read_csv(ROOT + '/data/train_discretize_oh.csv')

train_X = train.drop('Transported', axis=1)
Transported = train['Transported']
train_y = [1 if x == True else 0 for x in Transported]

test = pd.read_csv(ROOT + '/data/test_discretize_oh.csv')

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

print('--- FEATURE SELECTION ---')
from sklearn.feature_selection import SelectKBest, chi2
treshold = 0.7
best_k = train_X.shape[1]

for k in range(3,18): 
    sel = SelectKBest(chi2, k = k).fit(train_X, train_y)
    train_Xnorm_bestfeat = sel.transform(train_X)

    cv = cross_validate(
        estimator=MixedNB(categorical_features='all'),
        X = train_X, y = train_y, 
        return_train_score=True, 
        cv = 10, 
        n_jobs = -1, 
        verbose=0
    )
    score = np.mean(cv['test_score'])
    score_train = np.mean(cv['train_score'])
    
    print('Number of features most important selected, k = ', k)
    print('Result of the cv in test: ', score)
    print('Result of the cv in train: ', score_train)
    print('\n')
    
    if score >= treshold: 
        best_k = k 
        treshold = score
        best_train = score_train
        best_sel = sel
    
print('best_k: ', best_k)
print('Best score in test: ', treshold)
print('Best score in train:', best_train)