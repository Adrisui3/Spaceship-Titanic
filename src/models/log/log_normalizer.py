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

train_y = train_raw.Transported

cv = cross_validate(estimator = LogisticRegression(random_state = 1234, max_iter=1, solver = "newton-cholesky", C = 8, tol = 0.0001, fit_intercept=False, class_weight="balanced"), X = train_X, y = train_y, cv = 10, n_jobs = -1, verbose = 5)
print("Cross-validation test score: ", np.mean(cv["test_score"]))

#--------------- CREAR PREDICCIONES PARA EL TEST ---------------------------------

test_raw = utils.load_test()
test_raw["SM_FC"] = test_raw["ShoppingMall"] + test_raw["FoodCourt"]
test_raw["VD_SP"] = test_raw["Spa"] + test_raw["VRDeck"]
test_raw = test_raw.drop(["ShoppingMall", "FoodCourt", "Spa", "VRDeck"], axis=1)

test_raw_num = test_raw.drop(["PassengerId", "HomePlanet", "CryoSleep", "Destination", "VIP", "Cabin_deck", "Cabin_side"] , axis = 1)
test_raw_cat = test_raw.drop(["Age", "RoomService", "SM_FC", "VD_SP"], axis = 1)

# ------------------- SCALER ----------------------------
scaler = Normalizer()
test_scaled = pd.DataFrame(scaler.fit_transform(test_raw_num), columns=test_raw_num.columns)

# ----------------- CONCATENAR --------------------------

test_raw = pd.concat([test_raw_cat.reset_index(drop=True), test_scaled], axis=1)

test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))

print("Training LogisticRegression...")
# log = LogisticRegression(verbose=True, random_state=1234, max_iter = 1, C = 2, solver = "newton-cholesky", tol = 0.0001).fit(X = train_X, y = train_y)
# log = LogisticRegression(verbose=True, random_state=1234, max_iter = 1,solver = "newton-cholesky").fit(X = train_X, y = train_y)
# log = LogisticRegression(verbose=True, random_state=1234, max_iter = 1,solver = "newton-cholesky", fit_intercept=False, C = 1, tol = 0.0001).fit(X = train_X, y = train_y)
log = LogisticRegression(verbose=True, random_state = 1234, max_iter=1, solver = "newton-cholesky", C = 8, tol = 0.0001, fit_intercept=False, class_weight="balanced").fit(X = train_X, y = train_y)
print("Making predictions...")
pred_labels = log.predict(X = test)
print(pred_labels)

utils.generate_submission(labels = pred_labels, method = "log", notes = "norm-newton-chol-no-fit-i-balanced-weight")
