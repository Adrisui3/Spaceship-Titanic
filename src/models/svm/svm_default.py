import sys
sys.path.insert(1, './src')
from utils import *
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

train_raw = load_train()
train_X = one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train_y = train_raw.Transported

cv = cross_validate(estimator = SVC(random_state = 1234), X = train_X, y = train_y, cv = 10, n_jobs = -1, verbose = 5)
print("Cross-validation test score: ", np.mean(cv["test_score"]))

test_raw = load_test()
test = one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))

print("Training SVC...")
svc = SVC(verbose=True, random_state=1234).fit(X = train_X, y = train_y)
print("Making predictions...")
pred_labels = svc.predict(X = test)
print(pred_labels)

generate_submission(labels = pred_labels, method = "svm", notes = "default_parameters")