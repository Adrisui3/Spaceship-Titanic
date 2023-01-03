import sys
sys.path.insert(1, './src')
from utils import *
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate


train_raw = load_train()
train_x=train_raw.drop(['Transported', 'PassengerID'], axis=1)
train_y = train_raw.Transported

cv = cross_validate(estimator = DecisionTreeClassifier(), x = train_x, y = train_y,
                    cv = 10, n_jobs = -1, verbose = 5)
print(np.mean.cv(['test_score']))