import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate

train_raw = utils.load_train()
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train_X = utils.merge_numerical(df = train_X)
train_y = train_raw.Transported

cv = cross_validate(estimator = SVC(random_state = 1234), X = train_X, y = train_y, cv = 10, return_train_score = True, n_jobs = -1, verbose = 5)
print("Cross-validation train score: ", np.mean(cv["train_score"]))
print("Cross-validation test score: ", np.mean(cv["test_score"]))

test_raw = utils.load_test()
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))
test = utils.merge_numerical(df = test)

print("Training SVC...")
svc = SVC(verbose=True, random_state=1234).fit(X = train_X, y = train_y)
print(classification_report(train_y, svc.predict(X = train_X)))

print("Predicting for test...")
pred_labels = svc.predict(X = test)
#utils.generate_submission(labels = pred_labels, method = "svm", notes = "default_parameters")