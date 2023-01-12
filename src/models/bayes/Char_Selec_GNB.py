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

# # ------------------------------------------------------------------------------
# print('--- UNIVARIATE SELECTION ---')
# from sklearn.feature_selection import SelectKBest, chi2
# treshold = 0.7
# best_k = train_Xnorm.shape[1]

# for k in range(3,18): 
#     sel = SelectKBest(chi2, k = k).fit(train_Xnorm, train_y)
#     train_Xnorm_bestfeat = sel.transform(train_Xnorm)

#     cv = cross_validate(
#         estimator=GaussianNB(), 
#         X = train_Xnorm_bestfeat,  y = train_y, 
#         return_train_score = True, 
#         cv = 10, 
#         n_jobs=-1,
#         verbose=0
#     )
#     score = np.mean(cv['test_score'])
#     score_train = np.mean(cv['train_score'])
    
#     # print('Number of features most important selected, k = ', k)
#     # print('Result of the cv in test: ', score)
#     # print('Result of the cv in train: ', score_train)
#     # print('\n')
    
#     if score >= treshold: 
#         best_k = k 
#         treshold = score
#         best_train = score_train
#         best_sel = sel
    
# print('best_k: ', best_k)
# print('Best score in test: ', treshold)
# print('Best score in train:', best_train)

# train_Xnorm_bestfeat = best_sel.transform(train_Xnorm)
# test_norm_bestfeat = best_sel.transform(test_norm)

# gnb = GaussianNB().fit(train_Xnorm_bestfeat, train_y)
# pred_labels = gnb.predict(test_norm_bestfeat)

# # if treshold > 0.79 and best_train > 0.79: 
# #     utils.generate_submission(
# #         labels = pred_labels, 
# #         method = 'bayes', 
# #         notes = 'BestFeatSelection_Univariate_GNB_norm'
# #     )


# ------------------------------------------------------------------------------
# print('--- BACKWARD ELIMINATION ---')
# from sklearn.feature_selection import SequentialFeatureSelector
# treshold = 0.7
# best_k = train_Xnorm.shape[1]

# for k in range(3,17): 
#     sfs = SequentialFeatureSelector(
#         GaussianNB(), 
#         direction='backward',
#         n_features_to_select=k,
#         n_jobs=-1
#     ).fit(train_Xnorm, train_y)
    
#     train_Xnorm_bestfeat = sfs.transform(train_Xnorm)

#     cv = cross_validate(
#         estimator=GaussianNB(), 
#         X = train_Xnorm_bestfeat,  y = train_y, 
#         return_train_score = True, 
#         cv = 10, 
#         n_jobs=-1,
#         verbose=0
#     )
#     score = np.mean(cv['test_score'])
#     score_train = np.mean(cv['train_score'])
    
#     print('Number of features most important selected, k = ', k)
#     print('Result of the cv in test: ', score)
#     print('Result of the cv in train: ', score_train)
#     print('\n')
    
#     if score >= treshold: 
#         best_k = k 
#         treshold = score
#         best_train = score_train
#         best_sfs = sfs
    
# print('best_k: ', best_k)
# print('Best score in test: ', treshold)
# print('Best score in train:', best_train)

# train_Xnorm_bestfeat = best_sfs.transform(train_Xnorm)
# test_norm_bestfeat = best_sfs.transform(test_norm)

# gnb = GaussianNB().fit(train_Xnorm_bestfeat, train_y)
# pred_labels = gnb.predict(test_norm_bestfeat)

# if treshold > 0.79 and best_train > 0.79: 
#     utils.generate_submission(
#         labels = pred_labels, 
#         method = 'bayes', 
#         notes = 'BestFeatSelection_Backward_GNB_norm'
#     )
    
# ------------------------------------------------------------------------------
print('--- FORWARD ELIMINATION ---')
from sklearn.feature_selection import SequentialFeatureSelector
treshold = 0.7
best_k = train_Xnorm.shape[1]

for k in range(3,17): 
    sfs = SequentialFeatureSelector(
        GaussianNB(), 
        direction='forward',
        n_features_to_select=k,
        n_jobs=-1
    ).fit(train_Xnorm, train_y)
    
    train_Xnorm_bestfeat = sfs.transform(train_Xnorm)

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
    
    print('Number of features most important selected, k = ', k)
    print('Result of the cv in test: ', score)
    print('Result of the cv in train: ', score_train)
    print('\n')
    
    if score >= treshold: 
        best_k = k 
        treshold = score
        best_train = score_train
        best_sfs = sfs
    
print('best_k: ', best_k)
print('Best score in test: ', treshold)
print('Best score in train:', best_train)

train_Xnorm_bestfeat = best_sfs.transform(train_Xnorm)
test_norm_bestfeat = best_sfs.transform(test_norm)

gnb = GaussianNB().fit(train_Xnorm_bestfeat, train_y)
pred_labels = gnb.predict(test_norm_bestfeat)

# if treshold > 0.79 and best_train > 0.79: 
#     utils.generate_submission(
#         labels = pred_labels, 
#         method = 'bayes', 
#         notes = 'BestFeatSelection_Forward_GNB_norm'
#     )