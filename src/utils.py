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

def minmax_scaler_all_oh(df):    
    with open(file = ROOT + "/data/pickles/scalers/minmax_scaler_all_oh.pck", mode = "rb") as f:
        mmscaler = pickle.load(file = f)
        df = mmscaler.transform(df)
    return df

def robust_scaler_all_oh(df):    
    with open(file = ROOT + "/data/pickles/scalers/robust_scaler_all_oh.pck", mode = "rb") as f:
        rscaler = pickle.load(file = f)
        df = rscaler.transform(df)
    return df

def standard_scaler_all_oh(df):    
    with open(file = ROOT + "/data/pickles/scalers/standard_scaler_all_oh.pck", mode = "rb") as f:
        sscaler = pickle.load(file = f)
        df = sscaler.transform(df)
    return df

def normalizer_all_oh(df):    
    with open(file = ROOT + "/data/pickles/scalers/normalizer_all_oh.pck", mode = "rb") as f:
        norm = pickle.load(file = f)
        df = norm.transform(df)
    return df

def generate_submission(labels, method, notes = ""):
    test = pd.read_csv(ROOT + "/data/test_pr.csv")
    df = pd.DataFrame(test.PassengerId) 

    df["Transported"] = labels

    name = "/submissions/" + method + "/" + notes + "_" + str(datetime.datetime.now()).replace("-", "_")
    name = name.replace(" ", "_").replace(":", "-").replace(".", "-")
    df.to_csv(ROOT + name + ".csv", header = ['PassengerId', 'Transported'], index = False)