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
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import Normalizer
# Training data load and OneHotEncoder
print('--- LOADING DATA ---')
train_raw = utils.load_train()
train_X = utils.one_hot_encode(train_raw.drop(['Transported', 'PassengerId'], 
                                              axis = 1))

train_y = train_raw['Transported'].copy()

num_features = np.arange(0,6)
cat_features = np.arange(6,20)


print('--- NUMERICAL DATA ---')
train_Xnum = train_X.iloc[:, num_features].copy()
train_Xnum_norm =  Normalizer().fit_transform(train_Xnum)

gnb = GaussianNB().fit(X = train_Xnum_norm, y = train_y)
gaussian_log_proba = gnb.predict_log_proba(train_Xnum_norm)
print(gaussian_log_proba)

print('--- Categorical data ---')
train_Xcat = train_X.iloc[:, cat_features].copy()

mnb = MultinomialNB().fit(X = train_Xcat, y = train_y)
multinomial_log_proba = mnb.predict_log_proba(train_Xcat)
print(multinomial_log_proba)
print('\n', Normalizer().fit_transform(multinomial_log_proba))