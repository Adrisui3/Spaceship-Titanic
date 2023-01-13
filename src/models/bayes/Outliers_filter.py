# PATH for utils functions
import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

#Libraries 
import utils 
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB

print('--- LOAD DATA ---')
train_raw = utils.load_train()
train_X = utils.one_hot_encode(train_raw.drop(['Transported', 'PassengerId'], 
                                              axis = 1))

num_features = train_raw.select_dtypes(exclude=['object', 'bool']).columns
cat_features = train_X.drop(num_features, axis = 1).columns

print('--- DETECTING OUTLIERS --- ')
lof = LocalOutlierFactor(n_neighbors=20)
outliers = lof.fit_predict(train_X)
outliers = [i <= -0.5 for i in outliers]

train_raw['outlier'] = outliers
print('Number of outliers: ')
print(train_raw[train_raw['outlier'] == True]['outlier'].count())

train_X = utils.one_hot_encode(
    train_raw[train_raw['outlier'] == False].drop(['Transported', 'PassengerId', 'outlier'], axis = 1)
)
train_y = train_raw[train_raw['outlier'] == False]['Transported']


test_raw = utils.load_test()
test = utils.one_hot_encode(test_raw.drop(['PassengerId'], axis = 1))

print('\n--- NORMALIZER ---')
train_X_norm = train_X.copy()
train_X_norm.loc[:, num_features] = Normalizer().fit_transform(train_X_norm.loc[:, num_features])
test_norm = test.copy()
test_norm.loc[:, num_features] = Normalizer().fit_transform(test_norm.loc[:, num_features])

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

print('--- FEATURE SELECTION ---')
from sklearn.feature_selection import SelectKBest, chi2
treshold = 0.7
best_k = train_X_norm.shape[1]

for k in range(3,18): 
    sel = SelectKBest(chi2, k = k).fit(train_X_norm, train_y)
    train_Xnorm_bestfeat = sel.transform(train_X_norm)

    cv = cross_validate(
        estimator=GaussianNB(), 
        X = train_Xnorm_bestfeat,  y = train_y, 
        return_train_score = True, 
        cv = 10, 
        n_jobs=-1,
        verbose=0
    )
    score = np.mean(cv['test_score'])
    score_train = np.mean(cv['train_score'])
    
    # print('Number of features most important selected, k = ', k)
    # print('Result of the cv in test: ', score)
    # print('Result of the cv in train: ', score_train)
    # print('\n')
    
    if score >= treshold: 
        best_k = k 
        treshold = score
        best_train = score_train
        best_sel = sel
    
print('best_k: ', best_k)
print('Best score in test: ', treshold)
print('Best score in train:', best_train)

train_Xnorm_bestfeat = best_sel.transform(train_X_norm)
test_norm_bestfeat = best_sel.transform(test_norm)

gnb = GaussianNB().fit(train_Xnorm_bestfeat, train_y)
pred_labels = gnb.predict(test_norm_bestfeat)

utils.generate_submission(
    labels = pred_labels, 
    method = 'bayes', 
    notes = 'BestFeatSelection_Univariate_GNB_norm_outliersFilter'
)