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

minmax_scaler_all_oh = MinMaxScaler().fit(train)
with open(PATH + "minmax_scaler_all_oh.pck", "wb") as f:
    pickle.dump(minmax_scaler_all_oh, file = f)

robust_scaler_all_oh = RobustScaler().fit(train)
with open(PATH + "robust_scaler_all_oh.pck", "wb") as f:
    pickle.dump(robust_scaler_all_oh, file = f)

standard_scaler_all_oh = StandardScaler().fit(train)
with open(PATH + "standard_scaler_all_oh.pck", "wb") as f:
    pickle.dump(standard_scaler_all_oh, file = f)

normalizer_all_oh = Normalizer().fit(train)
with open(PATH + "normalizer_all_oh.pck", "wb") as f:
    pickle.dump(normalizer_all_oh, file = f)

