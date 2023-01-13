import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import ensemble
import utils
import math
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer


train_raw = pd.read_csv('/Users/marcosesquivelgonzalez/Desktop/Master C.Datos/Projects/MDatos_PreprocClasif/Spaceship-Titanic/data/train_pr_KnnImputed.csv')
train_raw = utils.merge_numerical(train_raw)
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))

RobScal = RobustScaler()
colnames = train_X.columns
df_array = RobScal.fit_transform(train_X)
train_X_RobScaled = pd.DataFrame(df_array,columns=colnames)

train_y = train_raw.Transported

params = {"n_neighbors":53, "p":2}
max_features = math.sqrt(train_X.shape[0]) / train_X.shape[0]
gsen = ensemble.BaggingClassifier(weak_estimator = KNeighborsClassifier(), n_estimators = 5000, estimator_params = params, verbose = True)
gsen.fit(X = train_X_RobScaled, y = train_y)
train_preds = gsen.predict(X = train_X_RobScaled)
print("Mean OOB accuracy:", gsen.get_mean_oob_accuracy())
print("Train score: ", accuracy_score(train_y, train_preds))

test_raw = pd.read_csv('/Users/marcosesquivelgonzalez/Desktop/Master C.Datos/Projects/MDatos_PreprocClasif/Spaceship-Titanic/data/test_pr_KnnImputed.csv')
test_raw = utils.merge_numerical(test_raw)
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))

colnames = test.columns
test_scaled_array = RobScal.transform(test)
test_scaled = pd.DataFrame(test_scaled_array,columns=colnames)

pred_labels = gsen.predict(X = test_scaled)
utils.generate_submission(labels = pred_labels, method = "ensemble", notes = "knn_euc_k52_merged_robscal_KnnImp_5000estimators")
