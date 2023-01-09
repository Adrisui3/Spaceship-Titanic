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
train_X = utils.one_hot_encode(train_raw.drop(['Transported', 'PassengerId'], 
                                              axis = 1))
train_y = train_raw['Transported']

print('\n--- NO SCALE ---')
cv = cross_validate(
    estimator=GaussianNB(), 
    X = train_X,  y = train_y, 
    return_train_score = True, 
    cv = 10, 
    n_jobs=-1,
    verbose=1 
)
print('Result of the cv in test: ', np.mean(cv['test_score']))
print('Result of the cv in train: ', np.mean(cv['train_score']))
# gnb = GaussianNB().fit(X = train_X, y = train_y)
# print(classification_report(train_y, gnb.predict(train_X)))

print('\n--- NORMALIZED ---')
train_X_norm = train_X.copy()
train_X_norm.iloc[:, 0:6] =  Normalizer().fit_transform(train_X.iloc[:,0:6])

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

print('\n--- MINMAX ---')
train_X_mm = train_X.copy()
train_X_mm.iloc[:, 0:6] =  Normalizer().fit_transform(train_X.iloc[:,0:6])

train_X_mm =  utils.minmax_scaler_oh(train_X, merged = False, nra = False)
cv = cross_validate(
    estimator=GaussianNB(), 
    X = train_X_mm,  y = train_y, 
    return_train_score = True, 
    cv = 10, 
    n_jobs=-1,
    verbose=1 
)
print('Result of the cv in test: ', np.mean(cv['test_score']))
print('Result of the cv in train: ', np.mean(cv['train_score']))
# gnb = GaussianNB().fit(X = train_X_mm, y = train_y)
# print(classification_report(train_y, gnb.predict(train_X_mm)))

print('\n--- ROBUST ---')
train_X_rob = train_X.copy()
train_X_rob.iloc[:, 0:6] =  Normalizer().fit_transform(train_X.iloc[:,0:6])

train_X_rob =  utils.minmax_scaler_oh(train_X, merged = False, nra = False)
cv = cross_validate(
    estimator=GaussianNB(), 
    X = train_X_rob,  y = train_y, 
    return_train_score = True, 
    cv = 10, 
    n_jobs=-1,
    verbose=1 
)
print('Result of the cv in test: ', np.mean(cv['test_score']))
print('Result of the cv in train: ', np.mean(cv['train_score']))
# gnb = GaussianNB().fit(X = train_X_rob, y = train_y)
# print(classification_report(train_y, gnb.predict(train_X_rob)))

print('\n--- STANDARDSCALER ---')
train_X_ss = train_X.copy()
train_X_ss.iloc[:, 0:6] =  Normalizer().fit_transform(train_X.iloc[:,0:6])

train_X_ss =  utils.standard_scaler_oh(train_X, merged = False, nra = False)
cv = cross_validate(
    estimator=GaussianNB(), 
    X = train_X_ss,  y = train_y, 
    return_train_score = True, 
    cv = 10, 
    n_jobs=-1,
    verbose=1 
)
print('Result of the cv in test: ', np.mean(cv['test_score']))
print('Result of the cv in train: ', np.mean(cv['train_score']))
# gnb = GaussianNB().fit(X = train_X_ss, y = train_y)
# print(classification_report(train_y, gnb.predict(train_X_ss)))
