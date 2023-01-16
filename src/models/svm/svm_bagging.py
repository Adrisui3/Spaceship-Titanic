import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import ensemble
import utils
import math
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score

train_raw = utils.load_train()
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train_X = utils.merge_numerical(df = train_X)
train_X = utils.standard_scaler_oh(df = train_X, merged = True)
train_y = train_raw.Transported

params = {'C': 8.832716109390496, 'gamma': 0.008999631421581993, "random_state":1234}
gsen = ensemble.BaggingClassifier(weak_estimator = SVC(), n_estimators = 100, estimator_params = params, verbose = True)

'''
print("--- CROSS VALIDATION ---")
cv = utils.stratified_cross_validation(estimator = gsen, X = train_X, y = train_y, verbose = True)
print("Cross-validation train score: ", np.mean(cv["train_score"]))
print("Cross-validation test score: ", np.mean(cv["test_score"]))
'''

gsen.fit(X = train_X, y = train_y)
train_preds = gsen.predict(X = train_X)
print("Train score: ", accuracy_score(train_y, train_preds))
print("Mean OOB accuracy:", gsen.get_mean_oob_accuracy())

test_raw = utils.load_test()
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))
test = utils.merge_numerical(df = test)
test = utils.standard_scaler_oh(df = test, merged=True)
pred_labels = gsen.predict(X = test)
utils.generate_submission(labels = pred_labels, method = "svm", notes = "svm_bagging_100_estimators")