import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import classification_report

train_raw = utils.load_train_KnnImp()
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train_X = utils.merge_numerical(df = train_X)
train_X = utils.standard_scaler_oh(df = train_X, merged=True)
train_y = train_raw.Transported

print("--- GRID SEARCH ---")
parameters = {'C': np.logspace(start = -2, stop = 10, base = 2),
              'gamma': np.logspace(start = -9, stop = 3, base = 2),
              'random_state':[1234]}

gs = GridSearchCV(estimator = SVC(), param_grid = parameters, cv = 10, return_train_score = True, n_jobs = -1, verbose = 3).fit(X = train_X, y = train_y)
print("Best score: ", gs.best_score_)
print("Best parameters: ", gs.best_params_)

print("--- CROSS VALIDATION ---")
cv = cross_validate(estimator = SVC().set_params(**gs.best_params_), X = train_X, y = train_y, cv = 10, return_train_score = True, n_jobs = -1)
print("Cross-validation train score: ", np.mean(cv["train_score"]))
print("Cross-validation test score: ", np.mean(cv["test_score"]))
svc = SVC().set_params(**gs.best_params_).fit(X = train_X, y = train_y)
print(classification_report(train_y, svc.predict(X = train_X)))

test_raw = utils.load_test()
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))
test = utils.merge_numerical(df = test)
test = utils.standard_scaler_oh(df = test, merged=True)
print("Predicting for test...")
pred_labels = svc.predict(X = test)
#utils.generate_submission(labels = pred_labels, method = "svm", notes = "gscv_standard_scaling_oh_logspace_knn_input")