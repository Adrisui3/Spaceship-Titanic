import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import ensemble
import utils
import math
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_raw = utils.load_train()
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train_y = train_raw.Transported

sammer = ensemble.SAMMERClassifier(weak_estimator = SVC(), n_estimators = 5)
sammer.fit(X = train_X.to_numpy(), y = train_y.to_numpy())