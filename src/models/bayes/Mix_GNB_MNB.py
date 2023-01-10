# PATH for utils functions
import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

#Libraries 
import utils 
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import Normalizer
from mixed_naive_bayes import MixedNB
from sklearn.preprocessing import LabelEncoder
# Training data load and OneHotEncoder
print('--- LOADING DATA ---')
train_raw = utils.load_train()
train_X = utils.one_hot_encode(train_raw.drop(['Transported', 'PassengerId'], 
                                              axis = 1))

num_features = np.arange(0,6)
cat_features = np.arange(6,20)

# Encoding target
train_y = train_raw['Transported']
le = LabelEncoder()
train_y = le.fit_transform(np.array(train_y))
# print(train_y)

print('--- NORMALIZER ---')
train_X.iloc[:,num_features] = Normalizer().fit_transform(train_X.iloc[:,num_features])
# print(train_X.iloc[:,num_features].describe())

print('--- CROSS VALIDATION ---')
# cross validation:
cv = cross_validate(
    estimator = MixedNB(categorical_features=list(cat_features)), 
    X = train_X,  y = train_y, 
    cv = 10, 
    return_train_score=True, 
    n_jobs=-1, # use all the cores
    verbose=0 # show in terminal the process
)
print('Result of the cv in test: ', np.mean(cv['test_score']))
print('Result of the cv in train: ', np.mean(cv['train_score']))

#Classification of the test data
test_raw = utils.load_test()
test = utils.one_hot_encode(test_raw.drop(['PassengerId'], axis = 1))

print('--- TEST ---')
MNB = MixedNB(categorical_features=list(cat_features)).fit(X = train_X, y = train_y)
pred_labels = MNB.predict(X = test)
pred_labels = np.bool_(pred_labels)
print(pred_labels)

utils.generate_submission(
    labels = pred_labels, 
    method = 'bayes', 
    notes = 'Mixed_Gaussian_Multinomial_NB'
)