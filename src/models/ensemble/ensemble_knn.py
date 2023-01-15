import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import ensemble
import utils
import math
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import RobustScaler


train_raw = utils.load_train_KnnImp()
train_raw = utils.merge_numerical(train_raw)
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))

RobScal = RobustScaler()
colnames = train_X.columns
df_array = RobScal.fit_transform(train_X)
train_X_RobScaled = pd.DataFrame(df_array,columns=colnames)

train_y = train_raw.Transported

params = {"n_neighbors":58, "metric":'correlation'}
max_features = math.sqrt(train_X.shape[0]) / train_X.shape[0]
gsen = ensemble.BaggingClassifier(weak_estimator = KNeighborsClassifier(), n_estimators = 5000, estimator_params = params, verbose = True)

gsen.fit(train_X_RobScaled,train_y)
#cv = utils.stratified_cross_validation(estimator = gsen, X = train_X, y = train_y,verbose=True)

#print(cv)

test_raw = utils.load_test_KnnImp()
test_raw = utils.merge_numerical(test_raw)
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))

colnames_tst = test.columns
test_scaled_array = RobScal.transform(test)
test_scaled = pd.DataFrame(test_scaled_array,columns=colnames_tst)

pred_labels = gsen.predict(X = test_scaled)
utils.generate_submission(labels = pred_labels, method = "ensemble", notes = "RobScal_k_58_correlation_merged_KnnImp_5000estimators")
