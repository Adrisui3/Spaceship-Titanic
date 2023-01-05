# PATH for utils functions
import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

#Libraries 
import utils 
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

# Training data load and OneHotEncoder
train_raw = utils.load_train()
train_X = utils.one_hot_encode(train_raw.drop(['Transported', 'PassengerId'], 
                                              axis = 1))
train_y = train_raw['Transported']

# GridSearch 
print('Grid Search...')
var_smooth_search = {'var_smoothing': np.logspace(0,-11, 100)}
GNB_grid = GridSearchCV(
    estimator = GaussianNB(), 
    param_grid=var_smooth_search,
    verbose=1, 
    n_jobs=-1, 
    cv=10
)
GNB_grid.fit(train_X, train_y)
print(GNB_grid.best_params_['var_smoothing'])

# Cross validation
print('Cross validation...')
cv = cross_validate(
    estimator = GaussianNB(var_smoothing=GNB_grid.best_params_['var_smoothing']), 
    X = train_X,  y = train_y, 
    cv = 10, 
    n_jobs=-1,
    verbose=1 
)
# print(cv)
print('Result of the cv: ', np.mean(cv['test_score']))

#Classification of the test data
test_raw = utils.load_test()
test = utils.one_hot_encode(test_raw.drop(['PassengerId'], axis = 1))

print('Training Gaussian NaiveBayes...')
GNB = GaussianNB().fit(X = train_X, y = train_y)
print('Classifying...')
pred_labels = GNB.predict(X = test)

utils.generate_submission(
    labels = pred_labels, 
    method = 'bayes', 
    notes = 'GridSearch_varsmoothing'
)