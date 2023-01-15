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
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score

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

# ------------------------------------------------------------------------------
print('\n--- UNIVARIATE SELECTION ---')
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

features_sel = best_sel.get_feature_names_out(best_sel.feature_names_in_)

train_Xnorm_bestfeat = pd.DataFrame(train_Xnorm_bestfeat, columns = features_sel)
test_norm_bestfeat = pd.DataFrame(test_norm_bestfeat, columns = features_sel)

# print(train_Xnorm.loc[:10, features_sel])
# print(train_Xnorm_bestfeat.head(10))

# ------------------------------------------------------------------------------
print('\n--- BAGGING ---')
n_estimators = 50000
ens = ensemble.BaggingClassifier(
    weak_estimator = GaussianNB(), 
    n_estimators = n_estimators, 
    estimator_params=None,
    verbose=True
).fit(X = train_Xnorm_bestfeat, y = train_y)
train_pred_labels = ens.predict(X = train_Xnorm_bestfeat)

print("Mean OOB accuracy:", ens.get_mean_oob_accuracy())
print("Train score: ", accuracy_score(train_y, train_pred_labels))

pred_labels = ens.predict(X = test_norm_bestfeat)
utils.generate_submission(
    labels = pred_labels, 
    method = 'ensemble', 
    notes = 'GNB_BestFeat_norm_' + str(n_estimators) + '_estimators'
)