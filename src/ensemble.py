import random
import numpy as np
from scipy.special import xlogy
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.base import clone

#####################################
# Bagging algorithm with soft voting#
#####################################

class BaggingClassifier:
    def __init__(self, weak_estimator, n_estimators, estimator_params = None, max_samples = 1.0, max_features = 1.0, verbose = False, random_state = None):
        self.__weak_estimator = weak_estimator
        self.__n_estimators = n_estimators
        self.__estimator_params = estimator_params
        self.__max_samples = max_samples
        self.__max_features = max_features
        self.__verbose = verbose
        self.__random_state = random_state
        self.__unfitted_estimators = [clone(self.__weak_estimator) for _ in range(self.__n_estimators)]
    
    def __bootstrap_samples(self, max_samples, n_samples, y):
        return resample(list(range(max_samples)), n_samples = n_samples, stratify = y)

    def __bootstrap_features(self, max_features, n_features):
        return random.sample(population = range(max_features), k = n_features)
    
    def __fit_estimators(self, X, y, n_samples, n_features):
        #max samples and max features
        max_samples, max_features = X.shape
        # Fit every estimator separately
        self.__fitted_estimators = []
        self.__estimator_features = []
        self.__oob_scores = []
        for i in range(self.__n_estimators):
            if self.__verbose:
                print("--- Fitting model", i, " ---")
            
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
            
            # If no parameters have been provided, use default
            params = self.__weak_estimator.get_params() if not self.__estimator_params else self.__estimator_params
            
            # Set fixed parameters
            if "probability" in self.__weak_estimator.get_params():
                params["probability"] = True
            
            # Fit weak estimator with either the best set of parameters or the default one
            estimator = self.__unfitted_estimators[i].set_params(**params).fit(X_bootstrap, y_bootstrap)
            self.__fitted_estimators.append(estimator)

            # Compute out of bag score
            oob_preds = estimator.predict(X = X_oob)
            self.__oob_scores.append(accuracy_score(y_oob, oob_preds))

        return

    def fit(self, X, y):        
        # Retrieve number of classes
        self.__classes = np.unique(y)

        # Find the number of samples used for bagging
        n_samples = max(1, int(X.shape[0] * self.__max_samples))
        # Find the number of features used for the random subspace
        n_features = max(1, int(X.shape[1] * self.__max_features))

        # Fit estimators
        self.__fit_estimators(X = X, y = y, n_samples = n_samples, n_features = n_features)
        
        # Compute weights for voting according to OOB scores
        self.__estimator_weights = np.array(self.__oob_scores) / sum(self.__oob_scores)

        return self

    def __compute_probas(self, X):
        return np.array([self.__fitted_estimators[i].predict_proba(X.iloc[:, self.__estimator_features[i]]) for i in range(self.__n_estimators)])

    def predict(self, X):
        # Predict probabilities
        probas = self.__compute_probas(X)
        probas_weighted = np.zeros(shape = (len(X), len(self.__classes)))
        
        # Weight predictions according to estimator's reliability
        for west, prob in zip(self.__estimator_weights, probas):
            probas_weighted += west * prob
        
        # For each observation, return the most likely label
        preds = []
        for probas in probas_weighted:
            preds.append(np.argmax(probas))

        return [self.__classes[pred_idx] for pred_idx in preds]

    def get_estimators(self):
        return self.__fitted_estimators
    
    def get_mean_oob_accuracy(self):
        return np.mean(self.__oob_scores)
    
    def get_estimator_weights(self):
        return self.__estimator_weights
    

#############################################################################################
# SAMME.R algorithm from J. Zhu, H. Zou, S. Rosset, T. Hastie. “Multi-class AdaBoost”, 2009.#
#############################################################################################

class SAMMERClassifier:
    def __init__(self, weak_estimator, n_estimators, estimator_params = None, learning_rate = 1.0, verbose = False):
        self.__n_estimators = n_estimators
        self.__weak_estimator = weak_estimator
        self.__estimator_params = estimator_params
        self.__learning_rate = learning_rate
        self.__verbose = verbose
    
    def __check_params(self):
        # If no parameters have been provided, use default
        self.__estimator_params = self.__weak_estimator.get_params() if not self.__estimator_params else self.__estimator_params

        # Set probability parameter if available
        if "probability" in self.__weak_estimator.get_params():
            self.__estimator_params["probability"] = True

    def fit(self, X, y):
        X, y = X.to_numpy(), y.to_numpy()

        # Check weak estimator parameters
        self.__check_params()

        # Retrieve number of classes
        unique_y = np.unique(y, return_inverse = True)
        self.__classes = unique_y[0]
        self.__K = len(self.__classes)

        # Label encode y
        enc_y = unique_y[1]

        # Uniformly initialize sample weights 
        sample_weights = np.full(X.shape[0], 1/X.shape[0])

        # Clean estimator if any
        self.__estimators = []
        
        # For every estimator...
        for m in range(self.__n_estimators):
            if self.__verbose:
                print("--- Fitting estimator ", m, " ---")
            
            # Create new estimator
            self.__estimators.append(clone(self.__weak_estimator))

            # Step 2.a -> Fit the estimator using current weights
            self.__estimators[m].set_params(**self.__estimator_params).fit(X = X, y = y, sample_weight = sample_weights)

            # Step 2.b -> Predict the weighted class probability estimates
            probas = self.__estimators[m].predict_proba(X = X)

            # If it's not the last estimator...
            if m < self.__n_estimators - 1:
                # Recodify labels as proposed in eq.(11)
                recod_y = np.full((X.shape[0], self.__K), -1/(self.__K - 1))
                for r, c in zip(range(X.shape[0]), enc_y):
                    recod_y[r, c] = 1.0
                
                # Step 2.d -> Update sample weights
                np.clip(probas, np.finfo(probas.dtype).eps, None, out=probas)
                sample_weights *= np.exp(-self.__learning_rate * ((self.__K - 1) / self.__K) * xlogy(recod_y, probas).sum(axis = 1))
                
                sum_weights = np.sum(sample_weights)
                if not np.isfinite(sum_weights) or sum_weights <= 0:
                    break

                # Step 2.e -> Normalize weights
                sample_weights /= np.sum(sample_weights)
        
        return self
    
    def __compute_probas(self, X):
        return np.array([est.predict_proba(X) for est in self.__estimators])

    def __compute_estimators_h(self, X):
        # Reset h values
        self.__estimator_h = []
        
        # Compute probabilities
        estimator_probas = self.__compute_probas(X = X)
        for probas in estimator_probas:
            # Step 2.c -> Compute the weight of the estimator.
            np.clip(probas, np.finfo(probas.dtype).eps, None, out=probas)
            probas_log = np.log(probas)
            self.__estimator_h.append((self.__K - 1) * (probas_log - (probas_log.sum(axis = 1)[:, np.newaxis] / self.__K)))

    def predict(self, X):
        X = X.to_numpy()
        
        # Compute estimator weights
        self.__compute_estimators_h(X = X)
        
        # Step 3. -> Compute outputs
        preds = sum(h for h in self.__estimator_h)
        
        # For each observation, return the most likely label
        idx_classes = []
        for pred in preds:
            idx_classes.append(np.argmax(pred))

        return [self.__classes[idx] for idx in idx_classes]
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X = X))
    
    def get_params(self, deep = False):
        return {"n_estimators":self.__n_estimators, "weak_estimator":self.__weak_estimator, 
                "estimator_params":self.__estimator_params, "learning_rate":self.__learning_rate}