import random
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

class GridSearchBaggingClassifier:
    def __init__(self, weak_estimator, n_estimators, estimator_params = None, max_samples = 1.0, max_features = 1.0, grid_search = True, verbose = False, random_state = 1234):
        self.__weak_estimator = weak_estimator
        self.__n_estimators = n_estimators
        self.__estimator_params = estimator_params
        self.__max_samples = max_samples
        self.__max_features = max_features
        self.__random_state = random_state
        self.__gs = grid_search
        self.__verbose = verbose
    
    def __bootstrap_samples(self, max_samples, n_samples, y):
        return resample(list(range(max_samples)), n_samples = n_samples, stratify = y)

    def __bootstrap_features(self, max_features, n_features):
        return random.sample(population = range(max_features), k = n_features)
    
    def __fit_estimators(self, X, y, params, n_samples, n_features):
        # Fix verbosity
        verbose = 3 if self.__verbose else 0
        #max samples and max features
        max_samples, max_features = X.shape
        # Fit every estimator separately
        self.__fitted_estimators = []
        self.__estimator_features = []
        self.__oob_scores = []
        for _ in range(self.__n_estimators):            
            if self.__verbose:
                print("--- Fitting model", _, " ---")
            
            # Subsample the rows
            samples_indices = self.__bootstrap_samples(max_samples, n_samples, y)
            oob_samples_indices = list(set(range(max_samples)) - set(samples_indices))
            # Subsample the columns
            features_indices = self.__bootstrap_features(max_features, n_features)
            self.__estimator_features.append(features_indices)
            
            # Sample new training set
            X_bootstrap, y_bootstrap = X.iloc[samples_indices, features_indices], y.iloc[samples_indices]
            # Sample out of bag set
            X_oob, y_oob = X.iloc[oob_samples_indices, features_indices], y.iloc[oob_samples_indices]
            
            # Check if grid search is required
            best_params = params
            if self.__gs:
                # Perform grid search with 10 fold stratified cross validation to find the best parameters
                gscv = GridSearchCV(estimator = self.__weak_estimator, param_grid = params, cv = 10, return_train_score = True, verbose = verbose, n_jobs = -1).fit(X = X_bootstrap, y = y_bootstrap)
                best_params = gscv.best_params_

            # Fit weak estimator with either the best set of parameters or the default one
            best_params["probability"] = True
            estimator = self.__weak_estimator.set_params(**best_params).fit(X_bootstrap, y_bootstrap)
            self.__fitted_estimators.append(estimator)

            # Compute out of bag score
            oob_preds = estimator.predict(X = X_oob)
            self.__oob_scores.append(accuracy_score(y_oob, oob_preds))

        return

    def fit(self, X, y):        
        # Find the number of samples used for bagging
        n_samples = max(1, int(X.shape[0] * self.__max_samples))
        # Find the number of features used for the random subspace
        n_features = max(1, int(X.shape[1] * self.__max_features))

        # If no parameters for GS have been specified, use default parameters
        if not self.__estimator_params:
            params = self.__weak_estimator.get_params()
        else:
            params = self.__estimator_params

        # Fit estimators
        self.__fit_estimators(X = X, y = y, params = params, n_samples = n_samples, n_features = n_features)
        
        # Compute weights for voting according to OOB scores
        self.__estimator_weights = np.array(self.__oob_scores) / sum(self.__oob_scores)

        return self

    def __compute_probas(self, X):
        return np.asarray([self.__fitted_estimators[i].predict_proba(X.iloc[:, self.__estimator_features[i]]) for i in range(self.__n_estimators)])

    def predict(self, X):
        probas = self.__compute_probas(X)


        return

    def get_estimators(self):
        return self.__fitted_estimators
    
    def get_oob_scores(self):
        return self.__oob_scores
    
    def get_estimator_weights(self):
        return self.__estimator_weights