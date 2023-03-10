import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import ensemble
import utils
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train_raw = utils.load_train()
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train_y = train_raw.Transported

params = {'criterion': 'gini','max_depth': 14,'max_features': 'sqrt','min_samples_leaf': 10,'min_samples_split': 60,'splitter': 'best'}
max_features = math.sqrt(train_X.shape[0]) / train_X.shape[0]
gsen = ensemble.BaggingClassifier(weak_estimator = DecisionTreeClassifier(), n_estimators = 256, estimator_params = params, verbose = True)
gsen.fit(X = train_X, y = train_y)
train_preds = gsen.predict(X = train_X)
print("Mean OOB accuracy:", gsen.get_mean_oob_accuracy())
print("Train score: ", accuracy_score(train_y, train_preds))

test_raw = utils.load_test()
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))
pred_labels = gsen.predict(X = test)
#utils.generate_submission(labels = pred_labels, method = "ensemble", notes = "decision_tree_test_max_features_depth5")
