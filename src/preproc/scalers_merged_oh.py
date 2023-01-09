import os
import sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(SRC)

import utils
import pickle
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Normalizer

PATH = utils.ROOT + "/data/pickles/scalers/"

train_raw = utils.load_train()
train = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train = utils.merge_numerical(df = train)

minmax_scaler = MinMaxScaler().fit(train)
with open(PATH + "minmax_scaler_merged_oh.pck", "wb") as f:
    pickle.dump(minmax_scaler, file = f)

robust_scaler = RobustScaler().fit(train)
with open(PATH + "robust_scaler_merged_oh.pck", "wb") as f:
    pickle.dump(robust_scaler, file = f)

standard_scaler = StandardScaler().fit(train)
with open(PATH + "standard_scaler_merged_oh.pck", "wb") as f:
    pickle.dump(standard_scaler, file = f)

normalizer = Normalizer().fit(train)
with open(PATH + "normalizer_merged_oh.pck", "wb") as f:
    pickle.dump(normalizer, file = f)