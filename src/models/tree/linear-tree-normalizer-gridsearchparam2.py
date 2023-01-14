import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)
import utils
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from lineartree import LinearTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import random

random.seed(1234)
# Cargamos conjunto de train
train_raw = utils.load_train()
train_x = utils.normalizer_oh(utils.one_hot_encode(df = train_raw.drop(['Transported', 'PassengerId'], axis=1)))
train_y = train_raw.Transported

train_x_numpy = train_x.to_numpy()
train_y_numpy = train_y.to_numpy()

# Creamos los folds
skf = StratifiedKFold(n_splits=10)

# Creamos el grid de parámetros
param_grid = {'criterion' : ["hamming"], 'max_depth': [5,10,15], 'min_samples_split': [12,15,18], 
              'min_samples_leaf':[55,60,65], 'max_bins':[20,25,30]}
grid = ParameterGrid(param_grid)


best_accuracy = 0
mean_train_accuracy = []
mean_test_accuracy = []
i = 1
# hacemos gridsearch
for params in grid:
    print('training '+str(i)+' of '+str(len(grid)))
    # Declaramos el árbol de decisión Lineal
    lt = LinearTreeClassifier(base_estimator=RidgeClassifier(random_state=1234), criterion=params['criterion'], max_depth=params['max_depth'],
                              min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'],
                              max_bins=params['max_bins'], n_jobs=-1)


    # entrenamos y evaluamos el modelo con validación cruzada
    accuracy_train = []
    accuracy_test = []
    for train, test in skf.split(train_x_numpy, train_y_numpy):
        lt.fit(train_x_numpy[train], train_y_numpy[train])
        pred_train = lt.predict(train_x_numpy[train])
        accuracy_train.append(accuracy_score(train_y_numpy[train], pred_train))
        pred_test = lt.predict(train_x_numpy[test])
        accuracy_test.append(accuracy_score(train_y_numpy[test], pred_test))

    mean_train_accuracy.append(np.mean(accuracy_train))
    mean_test_accuracy.append(np.mean(accuracy_test))
    i += 1
    if np.mean(accuracy_test)>best_accuracy:
        best_accuracy = np.mean(accuracy_test)
        best_params = params
# best train_accuracy
max(mean_train_accuracy)
# 0.821797991842522
# best_accuracy (test)
best_accuracy
# 0.8013367194423502
# best_accuracy (train)
mean_train_accuracy[mean_test_accuracy.index(best_accuracy)]
# 0.8116236743052511
# best_params
best_params
# {'criterion': 'hamming', 'max_bins': 30, 'max_depth': 5, 'min_samples_leaf': 65, 'min_samples_split': 12}
# Cambian los parámetros óptimos y aumenta la precisión


# Cargamos conjunto de test
test_raw = utils.load_test()
test = utils.normalizer_oh(utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1)))

# Predicciones
print("Training LinearTree classifier...")
lt = LinearTreeClassifier(base_estimator=RidgeClassifier(random_state=1234), criterion=best_params['criterion'], max_depth=best_params['max_depth'],
                          min_samples_split=best_params['min_samples_split'], min_samples_leaf=best_params['min_samples_leaf'],
                          max_bins=best_params['max_bins'], n_jobs=-1).fit(X = train_x_numpy, y = train_y_numpy)
print("Making predictions...")
pred_labels = lt.predict(X = test.to_numpy())

utils.generate_submission(labels = pred_labels, method = "tree", notes = "LinearTree_ridge_normalizer_gridsearch_parameters2")
