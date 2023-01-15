# PATH for utils functions
import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

#Libraries 
import ensemble 
import utils 
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_validate

print('--- LOAD AND NORMALIZER DATA ---')
train_raw = utils.load_train()
train_X = utils.one_hot_encode(train_raw.drop(['Transported', 'PassengerId'], 
                                              axis = 1))
train_y = train_raw['Transported']

num_features = train_raw.select_dtypes(exclude=['object', 'bool']).columns
cat_features = train_X.drop(num_features, axis = 1).columns

test_raw = utils.load_test()
test = utils.one_hot_encode(test_raw.drop(['PassengerId'], axis = 1))

#normalizer
train_Xnorm = train_X.copy()
train_Xnorm.loc[:, num_features] = Normalizer().fit_transform(train_Xnorm.loc[:, num_features])

test_norm = test.copy()
test_norm.loc[:, num_features] = Normalizer().fit_transform(test_norm.loc[:, num_features])

print('\n--- SAMMER ---')
sammer = ensemble.SAMMERClassifier(
    weak_estimator=GaussianNB(), 
    n_estimators=10, 
    learning_rate=0.1
).fit(train_Xnorm, train_y)

sammer.predict(test_norm)

# for lr in np.arange(0.01, 0.21, 0.01): 
#     sammer_Bayes = ensemble.SAMMERClassifier(
#         weak_estimator = GaussianNB(), 
#         n_estimators=50, 
#         estimator_params=None, 
#         learning_rate=lr, 
#         verbose=False
#     ).fit(train_Xnorm, train_y)
    
#     cv = cross_validate(
#         estimator=sammer_Bayes, 
#         X=train_Xnorm, y=train_y
#     )
    
#     print('Learning Rate: ', lr)
#     print('Train score: ', np.mean(cv['train_score']))
#     print('Test score: ', np.mean(cv['test_score']))