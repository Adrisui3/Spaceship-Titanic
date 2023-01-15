import os
import sys

MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
import ensemble
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

params = {'max_iter': 5, 'solver': 'newton-cholesky', 'C' : 1, 'tol' : 0.0001, 'fit_intercept' : False}

train_sc = []
test_sc = []
lr_list = []

for lr in np.arange(0.001,0.002, 0.001):
    gsen = ensemble.SAMMERClassifier(weak_estimator = LogisticRegression(verbose=False), n_estimators = 50, estimator_params = params, learning_rate = lr, verbose = False)
    cv = utils.stratified_cross_validation(estimator=gsen, X = train_X, y = train_y, verbose=False)

    print("--- LEARNING RATE:", lr)
    print("\tCross-validation train score: ", np.mean(cv["train_score"]))
    print("\tCross-validation test score: ", np.mean(cv["test_score"]))
    train_sc.append(np.mean(cv['train_score']))
    test_sc.append(np.mean(cv["test_score"]))
    lr_list.append(lr)
    np.mean(cv['test_score'])

print("RESULTADOS:")

print(test_sc)
print(train_sc)
print(lr_list)

# train_preds = gsen.predict(X = train_X)
# # print("Mean OOB accuracy:", gsen.get_mean_oob_accuracy())
# print("Train score: ", accuracy_score(train_y, train_preds))

# #--------------- CREAR PREDICCIONES PARA EL TEST ---------------------------------

# test_raw = utils.load_test()
# test = utils.one_hot_encode(df = test_raw.drop(['PassengerId'], axis=1))
# test = utils.merge_numerical(test)
# test_num = test.drop(["HomePlanet_Europa", "HomePlanet_Mars", "CryoSleep_1.0", "VIP_1.0", "Destination_PSO J318.5-22", "Destination_TRAPPIST-1e", "Cabin_deck_T", "Cabin_deck_B", "Cabin_deck_C", "Cabin_deck_D", "Cabin_deck_E", "Cabin_deck_F", "Cabin_deck_G", "Cabin_side_S"] , axis = 1)
# test_cat = test.drop(["Age", "RoomService", "SM_FC", "VD_SP"], axis = 1)

# test_num = utils.normalizer_oh(test_num, merged=True, nra=True, onlynum=True)
# test = pd.concat([test_cat.reset_index(drop=True), test_num], axis=1)

# print("Making predictions...")
# pred_labels = gsen.predict(X = test)
# print(pred_labels)

# # utils.generate_submission(labels = pred_labels, method = "log", notes = "bagging-nooutliers-10000")