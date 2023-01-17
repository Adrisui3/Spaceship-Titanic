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
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report

print('--- LOAD DATA ---')
train_raw = pd.read_csv(ROOT + '/data/train_pr_KnnImputed.csv')
train_X = utils.one_hot_encode(train_raw.drop(['Transported', 'PassengerId'], 
                                              axis = 1))
train_y = train_raw['Transported']

num_features = train_raw.select_dtypes(exclude=['object', 'bool']).columns
cat_features = train_X.drop(num_features, axis = 1).columns

test_raw = pd.read_csv(ROOT + '/data/test_pr_KnnImputed.csv')
test = utils.one_hot_encode(test_raw.drop(['PassengerId'], axis = 1))

# remove negative values
underzero = train_X[train_X['Spa'] < 0].index
train_X.iloc[underzero, 4] = 0
print(train_X.iloc[underzero, 4])

print('--- NORMALIZER ---')

train_Xnorm = train_X.copy()
train_Xnorm.loc[:, num_features] = Normalizer().fit_transform(train_Xnorm.loc[:, num_features])

test_norm = test.copy()
test_norm.loc[:, num_features] = Normalizer().fit_transform(test_norm.loc[:, num_features])

cv = cross_validate(
    estimator=GaussianNB(), 
    X = train_Xnorm,  y = train_y, 
    return_train_score = True, 
    cv = 10, 
    n_jobs=-1,
    verbose=1 
)
print('Result of the cv in test: ', np.mean(cv['test_score']))
print('Result of the cv in train: ', np.mean(cv['train_score']))

print('--- UNIVARIATE SELECTION ---')
from sklearn.feature_selection import SelectKBest, chi2
treshold = 0.7
best_k = train_Xnorm.shape[1]

for k in range(3,18): 
    sel = SelectKBest(chi2, k = k).fit(train_Xnorm, train_y)
    train_Xnorm_bestfeat = sel.transform(train_Xnorm)

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

train_Xnorm_bestfeat = best_sel.transform(train_Xnorm)
test_norm_bestfeat = best_sel.transform(test_norm)

gnb = GaussianNB().fit(train_Xnorm_bestfeat, train_y)
pred_labels = gnb.predict(test_norm_bestfeat)

utils.generate_submission(
    labels = pred_labels, 
    method = 'bayes', 
    notes = 'BestFeatSelection_Univariate_GNB_norm_KnnImputed'
)
