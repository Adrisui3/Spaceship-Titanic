import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)
import utils
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Cargamos conjunto de train
train_raw = utils.load_train()
train_x = utils.one_hot_encode(df = train_raw.drop(['Transported', 'PassengerId'], axis=1))
train_y = train_raw.Transported


# GridSerach para mejores parametros
grid_parameters = {'criterion':['gini', 'entropy', 'log_loss'], 'splitter':['best','random'], 'min_samples':list(range(2,11)),
              'min_samples_leaf':list(range(1,11)), 'max_features':['None', 'sqrt', 'log2']}

dtc_gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=1234), param_grid=grid_parameters,scoring='accuracy',
                      njobs=-1, cv=10, verbose=5, return_train_score=True)



# Cargamos conjunto de test
test_raw = utils.load_test()
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))

# Predicciones
print("Training DecisionTreeClassifier...")
dtc = DecisionTreeClassifier(random_state=1234).fit(X = train_x, y = train_y)
print("Making predictions...")
pred_labels = dtc.predict(X = test)
true_labels = utils.encode_labels(pred_labels)
utils.generate_submission(labels = true_labels, method = "tree", notes = "default_parameters")