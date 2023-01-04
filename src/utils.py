import pandas as pd
import os
import datetime
import pickle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def load_train():
    return pd.read_csv(ROOT + '/data/train_pr.csv')

def load_test():
    return pd.read_csv(ROOT + "/data/test_pr.csv")

def one_hot_encode(df):
    filenames = ["data/pickles/one_hot/oh_destination.pck", "data/pickles/one_hot/oh_home_planet.pck", 
                 "data/pickles/one_hot/oh_cryosleep.pck", "data/pickles/one_hot/oh_vip.pck", 
                 "data/pickles/one_hot/oh_cabin_deck.pck", "data/pickles/one_hot/oh_cabin_side.pck"]
    categorical_variables = ["Destination", "HomePlanet", "CryoSleep", "VIP", "Cabin_deck", "Cabin_side"]
    for filename, var in zip(filenames, categorical_variables):
        with open(file = ROOT+'/'+filename, mode = "rb") as f:
            oh_e = pickle.load(file = f)
            categories = [var + "_" + str(cat) for cat in oh_e.categories_[0]][1:]
            df[categories] = oh_e.transform(df[[var]])
            df = df.drop([var], axis = 1)
    return df

def generate_submission(labels, method, notes = ""):
    test = pd.read_csv(ROOT + "/data/test_pr.csv")
    df = pd.DataFrame(test.PassengerId) 

    df["Transported"] = labels

    name = "submissions/" + method + "/" + notes + "_" + str(datetime.datetime.now()).replace("-", "_")
    df.to_csv(name.replace(" ", "_").replace(":", "-").replace(".", "-") + ".csv", header = ['PassengerId', 'Transported'], index = False)