import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)
import utils
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

train_raw = utils.load_train()
train_x = utils.one_hot_encode(df = train_raw.drop(['Transported', 'PassengerId'], axis=1))
train_y = train_raw.Transported

train_x_numpy = train_x.to_numpy()
train_y_numpy = train_y.to_numpy()

# Creamos los folds
skf = StratifiedKFold(n_splits=10)

# Creamos el grid de parámetros
params = {'ccp_alpha': 0.0002, 'criterion': 'gini', 'max_depth': 16, 'max_features': 'sqrt', 'min_samples_leaf': 6, 'min_samples_split': 50,
          'splitter': 'best'}

best_accuracy = 0
mean_train_accuracy = []
mean_test_accuracy = []


dtc = DecisionTreeClassifier(ccp_alpha=params['ccp_alpha'],criterion=params['criterion'], max_depth=params['max_depth'], max_features=params['max_features'],
                             min_samples_leaf=params['min_samples_leaf'], min_samples_split=params['min_samples_split'], splitter=params['splitter'],
                             random_state=(1234))


# entrenamos y evaluamos el modelo con validación cruzada
accuracy_train = []
accuracy_test = []
for train, test in skf.split(train_x_numpy, train_y):
     dtc.fit(train_x_numpy[train], train_y_numpy[train])
     pred_train = dtc.predict(train_x_numpy[train])
     accuracy_train.append(accuracy_score(train_y_numpy[train], pred_train))
     pred_test = dtc.predict(train_x_numpy[test])
     accuracy_test.append(accuracy_score(train_y_numpy[test], pred_test))

mean_train_accuracy.append(np.mean(accuracy_train))
mean_test_accuracy.append(np.mean(accuracy_test))






