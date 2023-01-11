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
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import Normalizer
# Training data load and OneHotEncoder
print('--- LOADING DATA ---')
# TRAIN
train_raw = utils.load_train()
train_X = utils.one_hot_encode(train_raw.drop(['Transported', 'PassengerId'], 
                                              axis = 1))

train_y = train_raw['Transported'].copy()

num_features = np.arange(0,6)
cat_features = np.arange(6,20)

#TEST
test_raw = utils.load_test()
test = utils.one_hot_encode(test_raw.drop(['PassengerId'], axis = 1))


print('\n--- NUMERICAL DATA ---')
train_Xnum = train_X.iloc[:, num_features].copy()
train_Xnum_norm =  Normalizer().fit_transform(train_Xnum)

test_num = test.iloc[:, num_features].copy()
test_num_norm = Normalizer().fit_transform(test_num)

gnb = GaussianNB().fit(X = train_Xnum_norm, y = train_y)
gaussian_jll = gnb._joint_log_likelihood(test_num_norm)
# print('\n', gaussian_jll)
# print(gnb.class_prior_)


print('\n--- CATEGORICAL DATA ---')
train_Xcat = train_X.iloc[:, cat_features].copy()

test_cat = test.iloc[:, list(cat_features)].copy()
# print(cat_features)

cnb = CategoricalNB(force_alpha=True).fit(X = train_Xcat, y = train_y)
categorical_jll = cnb.predict_log_proba(test_cat) # not normalized
log_prior = cnb.class_log_prior_
# # print(np.exp(log_prior))


print('\n--- JOINING PROBABILITIES ---')
jlls = []
jlls.append(gaussian_jll)
jlls.append(categorical_jll)
jlls = np.hstack([jlls])

jll = jlls.sum(axis = 0)
proba = jll - log_prior # log probabilities
# print(proba)

#standarise data
num = np.exp(jll) 
div = np.sum(num, axis = 1, keepdims = True) 
proba_norm = num/div

pred_labels = np.bool_(np.argmax(proba_norm, axis = 1))
print(pred_labels)


utils.generate_submission(
    labels = pred_labels, 
    method = 'bayes', 
    notes = 'ManualMixed_Gaussian_Categorical_NB'
)