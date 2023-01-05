import os
import sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(SRC)

import utils
import pickle
from sklearn.preprocessing import MinMaxScaler