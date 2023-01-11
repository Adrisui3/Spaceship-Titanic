# PATH for utils functions
import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

#Libraries 
import utils 
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report

# Training data load and OneHotEncoder
train_raw = utils.load_train()
train_X = utils.merge_numerical(train_raw)
num_features = train_X.select_dtypes(exclude=['object', 'bool']).columns

train_X = utils.one_hot_encode(train_raw.drop(['Transported', 'PassengerId'], 
                                              axis = 1))
train_y = train_raw['Transported']

print('\n--- NORMALIZER ---')
train_X_norm = train_X.copy()
train_X_norm.loc[:, num_features] =  Normalizer().fit_transform(train_X.loc[:, num_features])
# print(train_X_norm.loc[:, num_features].describe())

cv = cross_validate(
    estimator=GaussianNB(), 
    X = train_X_norm,  y = train_y, 
    return_train_score = True, 
    cv = 10, 
    n_jobs=-1,
    verbose=1 
)
print('Result of the cv in test: ', np.mean(cv['test_score']))
print('Result of the cv in train: ', np.mean(cv['train_score']))
gnb = GaussianNB().fit(X = train_X_norm, y = train_y)
pred_labels = gnb.predict(train_X_norm)
# print(classification_report(train_y, gnb.predict(train_X_norm)))

# Testing Normalizer
test_raw = utils.load_test()
test = utils.merge_numerical(test_raw)
test = utils.one_hot_encode(test_raw.drop(['PassengerId'], axis = 1))

test_norm = test.copy()
test_norm.loc[:, num_features] =  Normalizer().fit_transform(test.loc[:, num_features])

GNB = GaussianNB().fit(X = train_X_norm, y = train_y)
print('Classifying...')
pred_labels = GNB.predict(X = test_norm)

utils.generate_submission(
    labels = pred_labels, 
    method = 'bayes', 
    notes = 'SumSomeNumerical_Normalized_GNB'
)

