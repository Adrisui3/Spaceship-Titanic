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


train_sc = []
test_sc = []
for lr in np.arange(0.01,0.21,0.01):
    dtc_ab = ensemble.SAMMERClassifier(weak_estimator = DecisionTreeClassifier(), n_estimators =50, estimator_params = parameters, learning_rate=lr ,verbose = True)
    cv = utils.stratified_cross_validation(estimator = dtc_ab, X = train_x, y = train_y, verbose = True)
    # resultados validación cruzada
    cv
    print("--- LEARNING RATE:", lr)
    print("\tCross-validation train score: ", np.mean(cv["train_score"]))
    print("\tCross-validation test score: ", np.mean(cv["test_score"]))
    train_sc.append(np.mean(cv['train_score']))
    test_sc.append(np.mean(cv["test_score"]))
    np.mean(cv['test_score'])



# Cargamos conjunto de test
test_raw = utils.load_test()
test = utils.normalizer_oh(utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1)))

# Predicciones
#print("Training DecisionTreeClassifier...")
#dtc_ab.fit(X = train_x, y = train_y)
#print("Making predictions...")
#pred_labels = dtc_ab.predict(X = test)
#true_labels = utils.encode_labels(pred_labels)
#utils.generate_submission(labels = true_labels, method = "tree", notes = "normalizer_bagging_100000_estimators")
