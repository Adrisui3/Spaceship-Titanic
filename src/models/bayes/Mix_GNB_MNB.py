# PATH for utils functions
import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

#Libraries 
import utils 
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from mixed_naive_bayes import MixedNB
from sklearn.preprocessing import LabelEncoder
# Training data load and OneHotEncoder
print('Loading data...')
train_raw = utils.load_train()
train_X = utils.one_hot_encode(train_raw.drop(['Transported', 'PassengerId'], 
                                              axis = 1))

# Encoding class
train_y = train_raw['Transported']
le = LabelEncoder()
train_y = le.fit_transform(np.array(train_y))

# Need to se which are the categorical features:
# print(train_X.columns[np.arange(6,20,1)]) # categorical: [6:20])

# Mixed Naive Bayes attempt
# print('Mixed Naive bayes over Train')
# MNB = MixedNB(categorical_features=np.arange(6,20,1)).fit(train_X, train_y)
# pred = MNB.predict(train_X)
# print(accuracy_score(train_y, pred))

# cross validation:
print('Cross validation with Mixed Naive Bayes...')
cv = cross_validate(
    estimator = MixedNB(categorical_features=np.arange(6,20,1)), 
    X = train_X,  y = train_y, 
    cv = 10, 
    n_jobs=-1, # use all the cores
    verbose=0 # show in terminal the process
)
print('Result of the cv: ', np.mean(cv['test_score']))

#Classification of the test data
test_raw = utils.load_test()
test = utils.one_hot_encode(test_raw.drop(['PassengerId'], axis = 1))

print('Training Gaussian NaiveBayes...')
MNB = MixedNB().fit(X = train_X, y = train_y)
print('Classifying...')
pred_labels = MNB.predict(X = test)
pred_labels = np.array(pred_labels, dtype = 'bool')

utils.generate_submission(
    labels = pred_labels, 
    method = 'bayes', 
    notes = 'Mixed_Gaussian_Multinomial_NB'
)