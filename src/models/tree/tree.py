import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)
import utils
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

# Cargamos conjunto de train
train_raw = utils.load_train()
train_x = utils.one_hot_encode(df = train_raw.drop(['Transported', 'PassengerId'], axis=1))
train_y = train_raw.Transported


# validación cruzada en train
cv = cross_validate(estimator = DecisionTreeClassifier(), X = train_x, y = train_y, cv = 10, n_jobs = -1, verbose = 5)
print(np.mean(cv['test_score']))


# Cargamos conjunto de test
test_raw = utils.load_test()
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))

# Predicciones
print("Training DecisionTreeClassifier...")
dtc = DecisionTreeClassifier(random_state=1234).fit(X = train_x, y = train_y)
print("Making predictions...")
pred_labels = dtc.predict(X = test)

utils.generate_submission(labels = pred_labels, method = "tree", notes = "default_parameters")