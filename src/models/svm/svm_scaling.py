import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report

train_raw = utils.load_train()
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train_y = train_raw.Transported

print("\n--- DEFAULT ---")
cv = cross_validate(estimator = SVC(random_state = 1234), X = train_X, y = train_y, cv = 10, return_train_score = True, n_jobs = -1)
print("Cross-validation train score: ", np.mean(cv["train_score"]))
print("Cross-validation test score: ", np.mean(cv["test_score"]))
svc = SVC(random_state = 1234).fit(X = train_X, y = train_y)
print(classification_report(train_y, svc.predict(X = train_X)))

print("\n--- MIN MAX SCALER ---")
train_X_mm = utils.minmax_scaler_all_oh(df = train_X)
cv = cross_validate(estimator = SVC(random_state = 1234), X = train_X_mm, y = train_y, cv = 10, return_train_score = True, n_jobs = -1)
print("Cross-validation train score: ", np.mean(cv["train_score"]))
print("Cross-validation test score: ", np.mean(cv["test_score"]))
svc_mm = SVC(random_state = 1234).fit(X = train_X_mm, y = train_y)
print(classification_report(train_y, svc_mm.predict(X = train_X_mm)))

print("\n--- STANDARD SCALER ---")
train_X_sc = utils.standard_scaler_all_oh(df = train_X)
cv = cross_validate(estimator = SVC(random_state = 1234), X = train_X_sc, y = train_y, cv = 10, return_train_score = True, n_jobs = -1)
print("Cross-validation train score: ", np.mean(cv["train_score"]))
print("Cross-validation test score: ", np.mean(cv["test_score"]))
svc_sc = SVC(random_state = 1234).fit(X = train_X_sc, y = train_y)
print(classification_report(train_y, svc_sc.predict(X = train_X_sc)))

test_raw = utils.load_test()
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))
test = utils.standard_scaler_all_oh(df = test)
print("Predicting for test...")
pred_labels = svc_sc.predict(X = test)
utils.generate_submission(labels = pred_labels, method = "svm", notes = "standard_scaling_oh")

print("\n--- ROBUST SCALER ---")
train_X_rs = utils.robust_scaler_all_oh(df = train_X)
cv = cross_validate(estimator = SVC(random_state = 1234), X = train_X_rs, y = train_y, cv = 10, return_train_score = True, n_jobs = -1)
print("Cross-validation train score: ", np.mean(cv["train_score"]))
print("Cross-validation test score: ", np.mean(cv["test_score"]))
svc_rs = SVC(random_state = 1234).fit(X = train_X_rs, y = train_y)
print(classification_report(train_y, svc_rs.predict(X = train_X_rs)))

print("\n--- NORMALIZER ---")
train_X_norm = utils.normalizer_all_oh(df = train_X)
cv = cross_validate(estimator = SVC(random_state = 1234), X = train_X_norm, y = train_y, cv = 10, return_train_score = True, n_jobs = -1)
print("Cross-validation train score: ", np.mean(cv["train_score"]))
print("Cross-validation test score: ", np.mean(cv["test_score"]))
svc_norm = SVC(random_state = 1234).fit(X = train_X_norm, y = train_y)
print(classification_report(train_y, svc_norm.predict(X = train_X_norm)))