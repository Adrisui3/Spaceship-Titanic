import os
import sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(SRC)

import utils
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
import pandas as pd

PATH = utils.ROOT + "/data/pickles/one_hot/"

train_raw = utils.load_train()
# train = train_raw.drop(["PassengerId"], axis = 1)

train_raw["SM_FC"] = train_raw["ShoppingMall"] + train_raw["FoodCourt"]
train_raw["VD_SP"] = train_raw["Spa"] + train_raw["VRDeck"]
train_raw = train_raw.drop(["ShoppingMall", "FoodCourt", "Spa", "VRDeck"], axis=1)

train_raw_num = train_raw.drop(["PassengerId", "HomePlanet", "CryoSleep", "Destination", "VIP", "Transported", "Cabin_deck", "Cabin_side"] , axis = 1)
train_raw_cat = train_raw.drop(["Age", "RoomService", "SM_FC", "VD_SP"], axis = 1)

# ------------------- SCALER ----------------------------
scaler = Normalizer()
train_scaled = pd.DataFrame(scaler.fit_transform(train_raw_num), columns=train_raw_num.columns)

# ----------------- CONCATENAR --------------------------

train = pd.concat([train_raw_cat.reset_index(drop=True), train_scaled], axis=1)

oh_destination = OneHotEncoder(categories = "auto", drop="first", sparse_output=False).fit(train[["Destination"]])
with open(PATH + "oh_destination.pck", "wb") as f:
    pickle.dump(oh_destination, file = f)

oh_home_planet = OneHotEncoder(categories = "auto", drop="first", sparse_output=False).fit(train[["HomePlanet"]])
with open(PATH + "oh_home_planet.pck", "wb") as f:
    pickle.dump(oh_home_planet, file = f)

oh_cryosleep = OneHotEncoder(categories = "auto", drop="if_binary", sparse_output=False).fit(train[["CryoSleep"]])
with open(PATH + "oh_cryosleep.pck", "wb") as f:
    pickle.dump(oh_cryosleep, file = f)

oh_vip = OneHotEncoder(categories = "auto", drop="if_binary", sparse_output=False).fit(train[["VIP"]])
with open(PATH + "oh_vip.pck", "wb") as f:
    pickle.dump(oh_vip, file = f)

oh_cabin_deck = OneHotEncoder(categories = "auto", drop="first", sparse_output=False).fit(train[["Cabin_deck"]])
with open(PATH + "oh_cabin_deck.pck", "wb") as f:
    pickle.dump(oh_cabin_deck, file = f)

oh_cabin_side = OneHotEncoder(categories = "auto", drop="if_binary", sparse_output=False).fit(train[["Cabin_side"]])
with open(PATH + "oh_cabin_side.pck", "wb") as f:
    pickle.dump(oh_cabin_side, file = f)
    