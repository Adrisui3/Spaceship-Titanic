import pandas as pd
import os
import datetime
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def load_train():
    return pd.read_csv(ROOT + '/data/train_pr.csv')

def load_train_KnnImp():
    return pd.read_csv(ROOT + '/data/train_pr_KnnImputed.csv')

def load_train_nooutliers():
    return pd.read_csv(ROOT + '/data/train_nooutliers.csv')

def load_test():
    return pd.read_csv(ROOT + "/data/test_pr.csv")

def load_test_KnnImp():
    return pd.read_csv(ROOT + "/data/test_pr_KnnImputed.csv")

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

def minmax_scaler_oh(df, merged = False, nra = False, onlynum = False):
    filename = ROOT + "/data/pickles/scalers/minmax_scaler_" + ("nra_" if nra else "")
    filename += "merged" if merged else "all"
    filename += "_onlynum_oh.pck" if (onlynum and merged and nra) else "_oh.pck"    
    with open(file = filename, mode = "rb") as f:
        mmscaler = pickle.load(file = f)
        df2 = mmscaler.transform(df)
    return pd.DataFrame(df2, columns=df.columns)

def robust_scaler_oh(df, merged = False, nra = False, onlynum = False):
    filename = ROOT + "/data/pickles/scalers/robust_scaler_" + ("nra_" if nra else "")
    filename += "merged" if merged else "all" 
    filename += "_onlynum_oh.pck" if (onlynum and merged and nra) else "_oh.pck"
    with open(file = filename, mode = "rb") as f:
        rscaler = pickle.load(file = f)
        df2 = rscaler.transform(df)
    return pd.DataFrame(df2, columns=df.columns)

def standard_scaler_oh(df, merged = False, nra = False, onlynum = False):
    filename = ROOT + "/data/pickles/scalers/standard_scaler_" + ("nra_" if nra else "")
    filename += "merged" if merged else "all" 
    filename += "_onlynum_oh.pck" if (onlynum and merged and nra) else "_oh.pck"
    with open(file = filename, mode = "rb") as f:
        sscaler = pickle.load(file = f)
        df2 = sscaler.transform(df)
    return pd.DataFrame(df2, columns=df.columns)

def normalizer_oh(df, merged = False, nra = False, onlynum = False):    
    filename = ROOT + "/data/pickles/scalers/normalizer_" + ("nra_" if nra else "")
    filename += "merged" if merged else "all" 
    filename += "_onlynum_oh.pck" if (onlynum and merged and nra) else "_oh.pck"
    with open(file = filename, mode = "rb") as f:
        norm = pickle.load(file = f)
        df2 = norm.transform(df)
    return pd.DataFrame(df2, columns=df.columns)

def generate_submission(labels, method, notes = ""):
    test = pd.read_csv(ROOT + "/data/test_pr.csv")
    df = pd.DataFrame(test.PassengerId) 

    df["Transported"] = labels

    name = "/submissions/" + method + "/" + notes + "_" + str(datetime.datetime.now()).replace("-", "_")
    name = name.replace(" ", "_").replace(":", "-").replace(".", "-")
    df.to_csv(ROOT + name + ".csv", header = ['PassengerId', 'Transported'], index = False)
    
def merge_numerical(df):
    df["SM_FC"] = df["ShoppingMall"] + df["FoodCourt"]
    df["VD_SP"] = df["VRDeck"] + df["Spa"]
    df = df.drop(["ShoppingMall", "FoodCourt", "VRDeck", "Spa"], axis = 1)

    return df

def encode_labels(labels):
    return ['True' if x==1 else 'False' for x in labels]


def stratified_cross_validation(estimator, X, y, n_folds = 10, random_state = None, verbose = False):
    cv = {"test_score":[], "train_score":[]}
    skf = StratifiedKFold(n_splits = n_folds, random_state = random_state)
    for i, (train_idx, test_idx) in enumerate(skf.split(X = X, y = y)):
        Xi_train, yi_train = X.loc[train_idx], y.loc[train_idx]
        Xi_test, yi_test = X.loc[test_idx], y.loc[test_idx]

        # Fit estimator
        est = estimator.fit(X = Xi_train, y = yi_train)

        # Predict
        train_preds = estimator.predict(X = Xi_train)
        acc_train = accuracy_score(yi_train, train_preds)
        test_preds = estimator.predict(X = Xi_test)
        acc_test = accuracy_score(yi_test, test_preds)

        # Store results
        cv["train_score"].append(acc_train)
        cv["test_score"].append(acc_test)
        
        # Print current state
        if verbose:
            print("Fold: [", i + 1, "/", n_folds,"] -> scores = (train =", acc_train, ", test=", acc_test, ")")

    return cv
