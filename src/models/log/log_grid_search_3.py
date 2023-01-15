import os
import sys

MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV as GS_CV
import pandas as pd

train_raw = utils.load_train_KnnImp()

train_X = utils.one_hot_encode(df = train_raw.drop(['Transported', 'PassengerId'], axis=1))
train_X = utils.merge_numerical(train_X)
train_X_num = train_X.drop(["HomePlanet_Europa", "HomePlanet_Mars", "CryoSleep_1.0", "VIP_1.0", "Destination_PSO J318.5-22", "Destination_TRAPPIST-1e", "Cabin_deck_T", "Cabin_deck_B", "Cabin_deck_C", "Cabin_deck_D", "Cabin_deck_E", "Cabin_deck_F", "Cabin_deck_G", "Cabin_side_S"] , axis = 1)
train_X_cat = train_X.drop(["Age", "RoomService", "SM_FC", "VD_SP"], axis = 1)

train_X_num = utils.normalizer_oh(train_X_num, merged=True, nra=True, onlynum=True)
train_X = pd.concat([train_X_cat.reset_index(drop=True), train_X_num], axis=1)

train_X = train_X.drop(["Cabin_deck_T", "HomePlanet_Mars", "VIP_1.0", "Destination_PSO J318.5-22"], axis=1) # 0.8038710104096399 53 y 23; 0.8039866142877928 con feat_selection hasta nombre muy largo

# print(train_X)

train_y = train_raw.Transported

# print(train_y)

log = LogisticRegression(random_state = 1234)

iter_range = list(range(0,5))

range_tol = [1e-4, 1e-3, 1e-2, 1e-1, 1]
range_C = list(range(1,15))

# pgrid = dict(max_iter=iter_range, solver = ["newton-cg"], tol = range_tol, C = range_C, fit_intercept = [False, True], class_weight = ["balanced", None])
pgrid = dict(max_iter=iter_range, solver = ["newton-cholesky"], tol = range_tol, C = range_C, fit_intercept = [False, True], class_weight = ["balanced", None])

grid = GS_CV(log, pgrid, scoring='accuracy', n_jobs = -1, cv =10, return_train_score=True)

grid_search = grid.fit(train_X,train_y)

print(grid_search.best_params_)

print(grid_search.best_score_)
