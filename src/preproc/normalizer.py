import os
import sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(SRC)

import utils
import pickle
from sklearn.preprocessing import Normalizer

PATH = utils.ROOT + "/data/pickles/norm/"

train_raw = utils.load_train()
train = train_raw.drop(["PassengerId"], axis = 1)

norm_age = Normalizer().fit(train[["Age"]])
with open(PATH + "norm_age.pck", "wb") as f:
    pickle.dump(norm_age, file = f)

norm_RoomService = Normalizer().fit(train[["RoomService"]])
with open(PATH + "norm_RoomService.pck", "wb") as f:
    pickle.dump(norm_RoomService, file = f)