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
from sklearn.preprocessing import Normalizer
import pandas as pd

train_raw = utils.load_train()

train_X = utils.one_hot_encode(df = train_raw.drop(['Transported', 'PassengerId'], axis=1))
train_X = utils.merge_numerical(train_X)
train_X_num = train_X.drop(["HomePlanet_Europa", "HomePlanet_Mars", "CryoSleep_1.0", "VIP_1.0", "Destination_PSO J318.5-22", "Destination_TRAPPIST-1e", "Cabin_deck_T", "Cabin_deck_B", "Cabin_deck_C", "Cabin_deck_D", "Cabin_deck_E", "Cabin_deck_F", "Cabin_deck_G", "Cabin_side_S"] , axis = 1)
train_X_cat = train_X.drop(["Age", "RoomService", "SM_FC", "VD_SP"], axis = 1)

train_X_num = utils.normalizer_oh(train_X_num, merged=True, nra=True, onlynum=True)
train_X = pd.concat([train_X_cat.reset_index(drop=True), train_X_num], axis=1)

# print(train_X)

train_y = train_raw.Transported

# print(train_y)

# Con robust_scaler_all_oh y max_iter = 1 obtenemos 0.7915599910056479
# Con standard_scaler_all_oh aun peor

# El clásico da max_iter = 50 y 0.7931730222345674

log = LogisticRegression(random_state = 1234)

iter_range = list(range(0,100))

range_tol = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
range_C = list(range(1,20))

pgrid = dict(max_iter=iter_range, solver = ["newton-cholesky"])

grid = GS_CV(log, pgrid, scoring='accuracy', n_jobs = -1, cv =10, return_train_score=True)

grid_search = grid.fit(train_X,train_y)

print(grid_search.best_params_)

print(grid_search.best_score_)
