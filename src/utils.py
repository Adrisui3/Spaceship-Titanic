import pandas as pd
import datetime
from sklearn.preprocessing import OneHotEncoder
import pickle

def load_data():
    return pd.read_csv("data/train_pr.csv")

def one_hot_encode(df):
    filenames = ["data/pickles/one_hot/oh_destination.pck", "data/pickles/one_hot/oh_home_planet.pck", 
                 "data/pickles/one_hot/oh_cryosleep.pck", "data/pickles/one_hot/oh_vip.pck", 
                 "data/pickles/one_hot/oh_cabin_deck.pck", "data/pickles/one_hot/oh_cabin_side.pck"]
    categorical_variables = ["Destination", "HomePlanet", "CryoSleep", "VIP", "Cabin_deck", "Cabin_side"]
    for filename, var in zip(filenames, categorical_variables):
        with open(file = filename, mode = "rb") as f:
            oh_e = pickle.load(file = f)



def generate_submission(labels, method, notes = ""):
    test = pd.read_csv("data/test_pr.csv")
    df = pd.DataFrame(test.PassengerId) 

    df["Transported"] = ["False" if p == 0 else "True" for p in labels]

    name = "submissions/" + method + "/" + notes + "_" + str(datetime.datetime.now()).replace("-", "_")
    df.to_csv(name.replace(" ", "_").replace(":", "-").replace(".", "-") + ".csv", header = ['PassengerId', 'Transported'], index = False)