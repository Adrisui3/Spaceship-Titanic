import os
import sys

MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

train_raw = utils.load_train()
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train_y = train_raw.Transported

cv = cross_validate(estimator = LogisticRegression(random_state = 1234), X = train_X, y = train_y, cv = 10, n_jobs = -1, verbose = 5)
print("Cross-validation test score: ", np.mean(cv["test_score"]))

test_raw = utils.load_test()
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))

print("Training LogisticRegression...")
log = LogisticRegression(verbose=True, random_state=1234).fit(X = train_X, y = train_y)
print("Making predictions...")
pred_labels = log.predict(X = test)
print(pred_labels)

utils.generate_submission(labels = pred_labels, method = "log", notes = "default_parameters")
