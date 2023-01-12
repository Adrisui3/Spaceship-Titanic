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
# train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))

# train_X = utils.standard_scaler_all_oh(df = train_X)

train_raw["SM_FC"] = train_raw["ShoppingMall"] + train_raw["FoodCourt"]
train_raw["VD_SP"] = train_raw["Spa"] + train_raw["VRDeck"]
train_raw = train_raw.drop(["ShoppingMall", "FoodCourt", "Spa", "VRDeck"], axis=1)

train_raw_num = train_raw.drop(["PassengerId", "HomePlanet", "CryoSleep", "Destination", "VIP", "Transported", "Cabin_deck", "Cabin_side"] , axis = 1)
train_raw_cat = train_raw.drop(["Age", "RoomService", "SM_FC", "VD_SP"], axis = 1)

# ------------------- SCALER ----------------------------
scaler = Normalizer()
train_scaled = pd.DataFrame(scaler.fit_transform(train_raw_num), columns=train_raw_num.columns)

# ----------------- CONCATENAR --------------------------

train_raw = pd.concat([train_raw_cat.reset_index(drop=True), train_scaled], axis=1)

# ----------------- ONE-HOT ENCODING --------------------
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))

# train_X = train_X.drop(["Age", "RoomService", "Cabin_deck_F", "Cabin_deck_G", "Cabin_deck_T", "Cabin_side_S"], axis=1)
# train_X = train_X.drop(["Destination_PSO J318.5-22", "Destination_TRAPPIST-1e", "Cabin_deck_B", "Cabin_deck_C", "Cabin_deck_D", "Cabin_deck_E"], axis=1)
# train_X = train_X.drop(["HomePlanet_Europa", "HomePlanet_Mars", "CryoSleep_1.0", "VIP_1.0"], axis=1)

# print(train_X)

train_y = train_raw.Transported

# El clásico da max_iter = 50 y 0.7931730222345674

log = LogisticRegression(random_state = 1234)

iter_range = list(range(0,30))

range_tol = [1e-4]
range_C = list(range(1,15))

pgrid = dict(max_iter=iter_range, solver = ["liblinear", "lbfgs", "newton-cholesky"], tol = range_tol, C = range_C, class_weight = ["balanced", None], fit_intercept = [True, False])

grid = GS_CV(log, pgrid, scoring='accuracy', n_jobs = -1, cv =10, return_train_score=True, verbose=True)

grid_search = grid.fit(train_X,train_y)

print(grid_search.best_params_)

print(grid_search.best_score_)
