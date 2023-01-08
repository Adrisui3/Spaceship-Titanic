import ensemble
import utils
from sklearn.svm import SVC

train_raw = utils.load_train()
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
train_y = train_raw.Transported

gsen = ensemble.GridSearchBaggingClassifier(weak_estimator = SVC(), n_estimators = 2, estimator_params = {"C":1, "gamma":0.2}, grid_search = False, verbose = True)
gsen.fit(X = train_X, y = train_y)
#gsen.predict(X = train_X)

print(gsen.get_oob_scores())
print(gsen.get_estimator_weights())

