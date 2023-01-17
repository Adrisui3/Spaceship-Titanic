import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)
import utils
import pandas as pd
import numpy as np
import six
from c45 import C45
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

# Cargamos conjunto de train
train_raw = utils.load_train()
train_x = utils.one_hot_encode(df = train_raw.drop(['Transported', 'PassengerId'], axis=1))
train_y = train_raw.Transported

train_x_numpy = train_x.to_numpy()
train_y_numpy = train_y.to_numpy()

# Creamos los folds
skf = StratifiedKFold(n_splits=10)

# Creamos el grid de parámetros
param_grid = {'prune': [False, True], 'gain_ratio' : [False, True], 'is_repeating': [False, True], 'max_depth': list(range(15,35)), 'min_samples_split': list(range(45,65))}
grid = ParameterGrid(param_grid)

best_accuracy = 0
mean_train_accuracy = []
mean_test_accuracy = []
i = 1
# hacemos gridsearch
for params in grid:
    print('training '+str(i)+' of '+str(len(grid)))
    # Declaramos el árbol de decisión id3
    id3 = Id3Estimator(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], prune=params['prune'],
                       gain_ratio=params['gain_ratio'], is_repeating=params['is_repeating'])


    # entrenamos y evaluamos el modelo con validación cruzada
    accuracy_train = []
    accuracy_test = []
    for train, test in skf.split(train_x_numpy, train_y_numpy):
        id3.fit(train_x_numpy[train], train_y_numpy[train])
        pred_train = id3.predict(train_x_numpy[train])
        accuracy_train.append(accuracy_score(train_y_numpy[train], pred_train))
        pred_test = id3.predict(train_x_numpy[test])
        accuracy_test.append(accuracy_score(train_y_numpy[test], pred_test))

    mean_train_accuracy.append(np.mean(accuracy_train))
    mean_test_accuracy.append(np.mean(accuracy_test))
    i += 1
    if np.mean(accuracy_test)>best_accuracy:
        best_accuracy = np.mean(accuracy_test)
        best_params = params



# Cargamos conjunto de test
test_raw = utils.load_test()
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))

# Predicciones
print("Training id3 classifier...")
id3 = Id3Estimator(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'], prune=best_params['prune'],
                   gain_ratio=best_params['gain_ratio'], is_repeating=best_params['is_repeating']).fit(X = train_x.to_numpy(), y = train_y.to_numpy())
print("Making predictions...")
pred_labels = id3.predict(X = test.to_numpy())

utils.generate_submission(labels = pred_labels, method = "tree", notes = "C45_gridsearch_parameters2")
