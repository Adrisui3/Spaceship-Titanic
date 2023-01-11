#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:31:32 2023

@author: marcosesquivelgonzalez
"""

import numpy as np
import TGA
import utils


train_raw = utils.load_train()
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))


trim_prop = 0.2

print(TGA._trimmed_mean(train_X,trim_prop))

print(np.apply_along_axis(TGA._trimmed_mean_1d, 0, train_X, k=int(trim_prop*len(train_X))))