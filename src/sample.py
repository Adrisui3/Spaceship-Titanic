import utils
import numpy as np

preds = np.random.randint(low = 0, high = 2, size = 4277)
utils.generate_submission(labels = preds, method = "svm", notes = "placeholder")
utils.generate_submission(labels = preds, method = "bayes", notes = "placeholder")
utils.generate_submission(labels = preds, method = "log", notes = "placeholder")
utils.generate_submission(labels = preds, method = "tree", notes = "placeholder")
utils.generate_submission(labels = preds, method = "knn", notes = "placeholder")