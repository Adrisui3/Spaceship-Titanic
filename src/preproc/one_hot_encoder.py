import os
import sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(SRC)

import utils
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

PATH = "data/pickles/one_hot/"

train_raw = utils.load_train()
train = train_raw.drop(["PassengerId"], axis = 1)

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