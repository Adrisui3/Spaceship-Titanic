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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

train_raw = utils.load_train()
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train_X = utils.merge_numerical(df = train_X)
train_X = utils.standard_scaler_oh(df = train_X, merged = True)
train_y = train_raw.Transported

params = {'C': 8.832716109390496, 'gamma': 0.008999631421581993, "probability":True, "random_state":1234}
sammer = ensemble.SAMMERClassifier(weak_estimator = SVC(), n_estimators = 50, estimator_params = params)

print("--- CROSS VALIDATION ---")
cv = utils.stratified_cross_validation(estimator = sammer, X = train_X, y = train_y)
print("Cross-validation results:", cv)