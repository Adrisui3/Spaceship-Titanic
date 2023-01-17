import os
import sys

MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV as GS_CV

train_raw = utils.load_train()

train_raw["SM_FC"] = train_raw["ShoppingMall"] + train_raw["FoodCourt"]
train_raw["VD_SP"] = train_raw["Spa"] + train_raw["VRDeck"]
train_raw = train_raw.drop(["ShoppingMall", "FoodCourt", "Spa", "VRDeck"], axis=1)
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))

train_y = train_raw.Transported

log = LogisticRegression(random_state = 1234)

iter_range = list(range(1,200))

range_tol = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
range_C = list(range(1,20))

pgrid = dict(max_iter=iter_range, solver=["liblinear"], tol = range_tol, C = range_C)

grid = GS_CV(log, pgrid, scoring='accuracy', n_jobs = -1, cv =10, return_train_score=True)

grid_search = grid.fit(train_X,train_y)

print(grid_search.best_params_)

print(grid_search.best_score_)
