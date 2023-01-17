import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)
import utils
import pandas as pd
import numpy as np
from c45 import C45
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Cargamos conjunto de train
train_raw = utils.load_train()
train_x = utils.one_hot_encode(df = train_raw.drop(['Transported', 'PassengerId'], axis=1))
train_y = train_raw.Transported

train_x_numpy = train_x.to_numpy()
train_y_numpy = train_y.to_numpy()

# Creamos los folds
skf = StratifiedKFold(n_splits=10)

# Declaramos el árbol de decisión id3
c_45 = C45(attrNames=train_x.columns.tolist())

# entrenamos y evaluamos el modelo con validación cruzada
accuracy_train = []
accuracy_test = []
for train, test in skf.split(train_x_numpy, train_y_numpy):
    c_45.fit(train_x_numpy[train], train_y_numpy[train])
    pred_train = c_45.predict(train_x_numpy[train])
    accuracy_train.append(accuracy_score(train_y_numpy[train], pred_train))
    pred_test = c_45.predict(train_x_numpy[test])
    accuracy_test.append(accuracy_score(train_y_numpy[test], pred_test))

np.mean(accuracy_train)
np.mean(accuracy_test)



# Cargamos conjunto de test
test_raw = utils.load_test()
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))

# Predicciones
print("Training id3 classifier...")
id3 = Id3Estimator().fit(X = train_x.to_numpy(), y = train_y.to_numpy())
print("Making predictions...")
pred_labels = id3.predict(X = test.to_numpy())

utils.generate_submission(labels = pred_labels, method = "tree", notes = "C45_default_parameters")
