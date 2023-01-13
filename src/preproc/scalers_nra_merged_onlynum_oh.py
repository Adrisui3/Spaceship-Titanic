import os
import sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(SRC)

import utils
import pickle
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Normalizer
import pandas as pd

PATH = utils.ROOT + "/data/pickles/scalers/"

train_raw = utils.load_train()
train = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train = utils.merge_numerical(df = train)

print(train)

train_raw_num = train.drop(["HomePlanet_Europa", "HomePlanet_Mars", "CryoSleep_1.0", "VIP_1.0", "Destination_PSO J318.5-22", "Destination_TRAPPIST-1e", "Cabin_deck_T", "Cabin_deck_B", "Cabin_deck_C", "Cabin_deck_D", "Cabin_deck_E", "Cabin_deck_F", "Cabin_deck_G", "Cabin_side_S"] , axis = 1)
train_raw_cat = train.drop(["Age", "RoomService", "SM_FC", "VD_SP"], axis = 1)

print(train_raw_cat)

minmax_scaler = MinMaxScaler().fit(train_raw_num)
with open(PATH + "minmax_scaler_nra_merged_onlynum_oh.pck", "wb") as f:
    pickle.dump(minmax_scaler, file = f)

robust_scaler = RobustScaler().fit(train_raw_num)
with open(PATH + "robust_scaler_nra_merged_onlynum_oh.pck", "wb") as f:
    pickle.dump(robust_scaler, file = f)

standard_scaler = StandardScaler().fit(train_raw_num)
with open(PATH + "standard_scaler_nra_merged_onlynum_oh.pck", "wb") as f:
    pickle.dump(standard_scaler, file = f)

normalizer = Normalizer().fit(train_raw_num)
with open(PATH + "normalizer_nra_merged_onlynum_oh.pck", "wb") as f:
    pickle.dump(normalizer, file = f)