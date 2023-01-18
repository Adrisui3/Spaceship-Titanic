import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)
import utils
import pandas as pd
import numpy as np
from sklearn import tree

import graphviz

# Cargamos conjunto de train
train_raw = utils.load_train()
train_x = utils.normalizer_oh(utils.one_hot_encode(df = train_raw.drop(['Transported', 'PassengerId'], axis=1)))
train_y = train_raw.Transported

best_parameters = {'ccp_alpha': 0.0004,'criterion': 'gini', 'max_depth': 14, 'max_features': 'sqrt', 'min_samples_leaf': 19, 'min_samples_split': 64,
                   'splitter': 'best'}

dtc = tree.DecisionTreeClassifier(criterion = best_parameters['criterion'], max_depth=best_parameters['max_depth'],max_features=best_parameters['max_features'],
                             min_samples_leaf=best_parameters['min_samples_leaf'], min_samples_split=best_parameters['min_samples_split'],
                             splitter=best_parameters['splitter'],ccp_alpha=best_parameters['ccp_alpha'],random_state=1234).fit(X = train_x, y = train_y)

dot_data = tree.export_graphviz(dtc, out_file=None, feature_names=train_x.columns, class_names='Transported', special_characters=True,filled=True)
graph = graphviz.Source(dot_data) 
graph.render("BestTree_SpaceshipTitanic")