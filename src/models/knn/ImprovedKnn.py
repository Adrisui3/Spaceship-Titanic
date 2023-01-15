#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: marcosesquivelgonzalez

Se implementa un nuevo knn basado en el paper:
    An Improved kNN Based on Class Contribution and Feature Weighting
    -HUANG Jie1,2, WEI Yongqing3,*,YI Jing2,4 and LIU Mengdi1,
    
"""

import os
import sys
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.abspath(os.path.join(MODELS, os.pardir))
sys.path.append(SRC)

import utils
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

#train_raw = pd.read_csv('/Users/marcosesquivelgonzalez/Desktop/Master C.Datos/Projects/MDatos_PreprocClasif/Spaceship-Titanic/data/train_pr_KnnImputed_RobScal.csv')
train_raw = utils.load_train_KnnImp()
train_raw = utils.merge_numerical(train_raw) 
train_y = train_raw.Transported
train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))
colnames_train = train_X.columns

#test_raw = pd.read_csv('/Users/marcosesquivelgonzalez/Desktop/Master C.Datos/Projects/MDatos_PreprocClasif/Spaceship-Titanic/data/test_pr_KnnImputed_RobScal.csv')
test_raw = utils.load_test_KnnImp()
test_raw = utils.merge_numerical(test_raw)
test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))
colnames_tst = test.columns

RobScal = Normalizer()
#RobScal = RobustScaler()

train_scaled_array = RobScal.fit_transform(train_X)
train_scaled = pd.DataFrame(train_scaled_array,columns=train_X.columns)

test_scaled_array = RobScal.transform(test)
test_scaled = pd.DataFrame(test_scaled_array,columns=test.columns)

#-----------------------Calculo la ponderación de cada atributo:---------------------------

knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn7 = KNeighborsClassifier(n_neighbors=7)

score_cv_3 = cross_validate(estimator = knn3, X = train_scaled,
                          y = train_y, cv = 10, n_jobs = -1)
mean_score_3 = np.mean(score_cv_3['test_score'])

score_cv_5 = cross_validate(estimator = knn5, X = train_scaled,
                          y = train_y, cv = 10, n_jobs = -1)
mean_score_5 = np.mean(score_cv_5['test_score'])

score_cv_7 = cross_validate(estimator = knn7, X = train_scaled,
                          y = train_y, cv = 10, n_jobs = -1)
mean_score_7 = np.mean(score_cv_7['test_score'])

SCV_total = np.mean([mean_score_3,mean_score_5,mean_score_7])


SCV_i = np.zeros(train_scaled.shape[1])

for n_i,i in enumerate(train_scaled.columns):
    score_cv_3 = cross_validate(estimator = knn3, X = train_scaled.drop(i,axis=1),
                              y = train_y, cv = 10, n_jobs = -1)
    mean_score_3 = np.mean(score_cv_3['test_score'])
    
    score_cv_5 = cross_validate(estimator = knn5, X = train_scaled.drop(i,axis=1),
                              y = train_y, cv = 10, n_jobs = -1)
    mean_score_5 = np.mean(score_cv_5['test_score'])
    
    score_cv_7 = cross_validate(estimator = knn7, X = train_scaled.drop(i,axis=1),
                              y = train_y, cv = 10, n_jobs = -1)
    mean_score_7 = np.mean(score_cv_7['test_score'])
    
    SCV_i[n_i] = np.mean([mean_score_3,mean_score_5,mean_score_7])
        

Disc_i = 1 - (SCV_i - SCV_total)

Disc_i_norm = Disc_i/np.sum(Disc_i)#Estos serán los pesos a usar en knn


#------------------------------Algoritmo de knn:-----------------------------------------------



def improved_knn(n_neighbors,train,train_classes,test,weights=Disc_i_norm):
    
    Transported = train_classes
    train_data = np.array(train)
    test_data = np.array(test)
    
    distances = pairwise_distances(train_data,test_data,metric='minkowski',p=2,w=weights)
    
    #Los indices qué indicarán las posiciones de las instancias de train más cercanas
    #ind= np.zeros((len(test_data),n_neighbors))
    predicted = np.zeros(distances.shape[1])
    #Para cada instancia 'j' del test busco los k vecinos más cercanos
    for j in range(distances.shape[1]):
        
        j_dist = distances[:,j].copy()
        j_dist.sort()      #ordeno distancias de menor a mayor
        
        kminor_j_dist =  j_dist[:n_neighbors] #cojo las k distancias más pequeñas
        j_indeces = np.where(np.isin(distances[:,j],kminor_j_dist))[0][:n_neighbors] #extraigo los índices para esas distancias
        #No están ordenados acorde de menor a mayor distancia
        #Las ordeno:
        j_ind = j_indeces[np.argsort(distances[j_indeces,j])]   #Representa los índices de las instancias de train más pegadas al ejemplo j

        j_targets = Transported.iloc[j_ind]
        
        
        number_TrueLabels = np.sum(j_targets==True)
        number_FalseLabels = np.sum(j_targets==False)
        
        if number_TrueLabels == 0:
            predicted[j] = 0
            
        elif number_TrueLabels == n_neighbors:
            predicted[j] = 1
            
        else:              
            #Para la clase TRUE:
            CT_True = n_neighbors/number_TrueLabels + np.mean( kminor_j_dist[j_targets==True] )/number_TrueLabels
            
            #Para la clase FALSE:
            CT_False = n_neighbors/number_FalseLabels + np.mean( kminor_j_dist[j_targets==False] )/number_FalseLabels
        
            if CT_True < CT_False:
                predicted[j] = 1
            else:
                predicted[j] = 0
            
            
    return predicted

#-------------------------------------------------------------------------------------

#--------------Probamos si se comporta parecido al knn de sklearn como checkeo:
"""
prediction = improved_knn(n_neighbors=30, train=train_scaled, train_classes=train_y, test=test_scaled)

#knn = KNeighborsClassifier(n_neighbors=56)
#knn.fit(train_scaled,train_y)
#prediction2 = knn.predict(test_scaled)

#print(np.mean(prediction==prediction2))
         
#Tiene unos resultados parecidos, parece bien implementado

#---------------_Submission con el mejor valor de k -------------------

predicted_labels = utils.encode_labels(prediction)
utils.generate_submission(labels = predicted_labels, method = "knn", notes = "RobScal_k_49_ImprovedKnn_merged_knnImp") 
"""