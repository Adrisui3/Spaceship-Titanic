import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)
import ensemble
import utils
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Cargamos conjunto de train
train_raw = utils.load_train()
train_x = utils.normalizer_oh(utils.one_hot_encode(df = train_raw.drop(['Transported', 'PassengerId'], axis=1)))
train_y = train_raw.Transported


# GridSerach para mejores parametros
parameters = {'criterion':'gini', 'splitter':'best', 'max_depth':14, 'min_samples_split':64, 
                   'min_samples_leaf':19, 'max_features':'sqrt',  
                   'ccp_alpha':0.0004, 'random_state':1234}

dtc_bg = ensemble.BaggingClassifier(weak_estimator = DecisionTreeClassifier(), n_estimators =100000 , estimator_params = parameters, verbose = True)

# realizamos validación cruzada
cv = utils.stratified_cross_validation(estimator = dtc_bg, X = train_x, y = train_y, verbose = True)

# resultados validación cruzada
cv
np.mean(cv['train_score'])
# 64 estimators: 0.821797968969378
# 100000 estimators: 0.8224114773384652

np.mean(cv['test_score'])
# 64 estimators: 0.8011094797825484
# 100000 estimators: 0.8009941404441623


# Cargamos conjunto de test
test_raw = utils.load_test()
test = utils.normalizer_oh(utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1)))

# Predicciones
print("Training DecisionTreeClassifier...")
dtc_bg.fit(X = train_x, y = train_y)
print("Making predictions...")
pred_labels = dtc_bg.predict(X = test)
true_labels = utils.encode_labels(pred_labels)
utils.generate_submission(labels = true_labels, method = "tree", notes = "normalizer_bagging_100000_estimators")
