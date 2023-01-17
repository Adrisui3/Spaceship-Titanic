import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import ensemble
import utils
import math
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_raw = utils.load_train()
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train_X = utils.merge_numerical(df = train_X)
train_X = utils.standard_scaler_oh(df = train_X, merged = True)
train_y = train_raw.Transported

params = {'C': 8.832716109390496, 'gamma': 0.008999631421581993, "random_state":1234}

'''
learning_rates = np.arange(start=1.0, stop=2.1, step=0.1)
for lr in learning_rates:
    print("--- LEARNING RATE:", lr)
    sammer = ensemble.SAMMERClassifier(weak_estimator = SVC(), n_estimators = 15, estimator_params = params, learning_rate = lr)
    cv = utils.stratified_cross_validation(estimator = sammer, X = train_X, y = train_y)
    print("\tCross-validation train score: ", np.mean(cv["train_score"]))
    print("\tCross-validation test score: ", np.mean(cv["test_score"]))
'''

sammer = ensemble.SAMMERClassifier(weak_estimator = SVC(), n_estimators = 15, estimator_params = params, learning_rate = 0.3, verbose = True)
sammer.fit(X = train_X, y = train_y)
train_preds = sammer.predict(X = train_X)
print("Train score: ", accuracy_score(train_y, train_preds))

test_raw = utils.load_test()
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))
test = utils.merge_numerical(df = test)
test = utils.standard_scaler_oh(df = test, merged=True)
pred_labels = sammer.predict(X = test)
utils.generate_submission(labels = pred_labels, method = "svm", notes = "svm_adaboost_15_estimators_lr_gs")