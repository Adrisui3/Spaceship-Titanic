import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, cross_validate

train_raw = utils.load_train()
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train_X = utils.merge_numerical(df = train_X)
train_X = utils.standard_scaler_oh(df = train_X, merged = True)
train_y = train_raw.Transported

sfs = SequentialFeatureSelector(estimator = SVC(random_state=1234), n_features_to_select = "auto", tol = 0.0020, direction = "forward", cv = 10, n_jobs = -1).fit(X = train_X, y = train_y)
train_X_forw = sfs.transform(X = train_X)
print("Initial features: ", train_X.shape)
print("Selected features: ", train_X_forw.shape)

print("--- CROSS VALIDATION FEATURE SELECTION ---")
cv = cross_validate(estimator = SVC(random_state=1234), X = train_X_forw, y = train_y, cv = 10, return_train_score = True, n_jobs = -1)
print("Cross-validation train score: ", np.mean(cv["train_score"]))
print("Cross-validation test score: ", np.mean(cv["test_score"]))

print("--- GRID SEARCH ---")
parameters = {'C': np.logspace(start = -2, stop = 10, base = 2),
              'gamma': np.logspace(start = -9, stop = 3, base = 2),
              'random_state':[1234]}

gs = GridSearchCV(estimator = SVC(), param_grid = parameters, cv = 10, return_train_score = True, n_jobs = -1, verbose = 3).fit(X = train_X_forw, y = train_y)
print("Best score: ", gs.best_score_)
print("Best parameters: ", gs.best_params_)
print("--- CROSS VALIDATION GRID SEARCH + FEATURE SELECTION---")
cv_gs = cross_validate(estimator = SVC().set_params(**gs.best_params_), X = train_X_forw, y = train_y, cv = 10, return_train_score = True, n_jobs = -1)
print("Cross-validation train score: ", np.mean(cv_gs["train_score"]))
print("Cross-validation test score: ", np.mean(cv_gs["test_score"]))
svc_fs = SVC().set_params(**gs.best_params_).fit(X = train_X_forw, y = train_y)
print(classification_report(train_y, svc_fs.predict(X = train_X_forw)))

test_raw = utils.load_test()
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))
test = utils.merge_numerical(df = test)
test = utils.standard_scaler_oh(df = test, merged=True)
test = sfs.transform(X = test)
print("Predicting for test...")
pred_labels = svc_fs.predict(X = test)
utils.generate_submission(labels = pred_labels, method = "svm", notes = "svm_forward_sfs_4features")