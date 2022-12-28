import pandas as pd
import datetime

def load_data():
    return pd.read_csv("data/train.csv")

def generate_submission(labels, method, notes = ""):
    test = pd.read_csv("data/test.csv")
    df = pd.DataFrame(test.PassengerId) 

    df["Transported"] = ["False" if p == 0 else "True" for p in labels]

    name = "submissions/" + method + "/" + notes + "_" + str(datetime.datetime.now()).replace("-", "_")
    df.to_csv(name.replace(" ", "_").replace(":", "-").replace(".", "-") + ".csv", header = ['PassengerId', 'Transported'], index = False)