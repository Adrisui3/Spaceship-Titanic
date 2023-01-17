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
train_raw = utils.load_train_KnnImp()
train_x = utils.normalizer_oh(utils.one_hot_encode(df = train_raw.drop(['Transported', 'PassengerId'], axis=1)))
train_y = train_raw.Transported


# GridSerach para mejores parametros
grid_parameters = {'criterion':['gini'], 'splitter':['best'], 'max_depth':list(range(10,25)), 'min_samples_split':list(range(30,60)), 
                   'min_samples_leaf':list(range(2,20)), 'max_features':['sqrt'],  
                   'ccp_alpha':[0.00043,0.00044,0.00045,0.00046,0.00047]}

dtc_gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=1234), param_grid=grid_parameters,scoring='accuracy',
                      n_jobs=-1, cv=10, verbose=3, return_train_score=True)

dtc_gs.fit(X = train_x, y = train_y)
dtc_gs.best_score_
# mejor resultado con 0.7994994907609486
dtc_gs.best_params_
# {'ccp_alpha': 0.00045,
# 'criterion': 'gini',
# 'max_depth': 17,
# 'max_features': 'sqrt',
# 'min_samples_leaf': 8,
# 'min_samples_split': 37,
# 'splitter': 'best'}
# mejora, pruebo subirlo

best_parameters = dtc_gs.best_params_
# Cargamos conjunto de test
test_raw = utils.load_test_KnnImp()
test = utils.normalizer_oh(utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1)))

# Predicciones
print("Training DecisionTreeClassifier...")
dtc = DecisionTreeClassifier(criterion = best_parameters['criterion'], max_depth=best_parameters['max_depth'],max_features=best_parameters['max_features'],
                             min_samples_leaf=best_parameters['min_samples_leaf'], min_samples_split=best_parameters['min_samples_split'],
                             splitter=best_parameters['splitter'],ccp_alpha=best_parameters['ccp_alpha'],random_state=1234).fit(X = train_x, y = train_y)
print("Making predictions...")
pred_labels = dtc.predict(X = test)
true_labels = utils.encode_labels(pred_labels)
utils.generate_submission(labels = true_labels, method = "tree", notes = "normalizer_knnImp_gridSearch_parameters5")
