#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 00:27:16 2023

@author: marcosesquivelgonzalez
"""
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

train_raw  = pd.read_csv(ROOT + '/data/train_pr_to_KnnImp.csv')

Transported = train_raw.Transported
passID = train_raw[['PassengerId']]
train_raw.drop(['PassengerId','Name','Cabin_num'],axis=1,inplace=True)


test_raw = pd.read_csv(ROOT + '/data/test_pr_to_KnnImp.csv')

passIdtest = test_raw[['PassengerId']]
test_raw.drop(['PassengerId','Name','Cabin_num'],axis=1,inplace=True)

#------------ Codifico sin passenger Id ------------------------------------------
train_raw.info()

test_raw.info()

train_encoded = train_raw.copy()
test_encoded = test_raw.copy()

cat_columns = test_raw.select_dtypes(include=['object']).columns#Así no cojo transported

enc = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan)
enc.fit(train_raw[cat_columns])

train_encoded[cat_columns] = enc.transform(train_raw[cat_columns])
test_encoded[cat_columns] = enc.transform(test_raw[cat_columns])

#-----------------Standarizo e imputo SOLO con train---------------------------------------------------------
colnames = train_encoded.columns


RobScal = RobustScaler()

train_EncScal_array = RobScal.fit_transform(train_encoded)
train_EncScal = pd.DataFrame(train_EncScal_array,columns=colnames)

imputer = KNNImputer(n_neighbors=30)

train_EncScalImp = imputer.fit_transform(train_EncScal)
train_EncScalImp = pd.DataFrame(train_EncScalImp,columns=colnames)

#---------------------Quito estandarizacion y codificación-----------------------------
train_EncImp_array = RobScal.inverse_transform(train_EncScalImp)
train_EncImp = pd.DataFrame(train_EncImp_array,columns=colnames)


train_Imp = train_EncImp.copy()
train_Imp[cat_columns] = enc.inverse_transform(train_EncImp[cat_columns])

#---------------------------Pasamos ahora a estandarizar e imputar para test------------------------------------------

#Cogemos los datos de train imputados y codificados y le quitamos transported_
train_encoded_forTst = train_EncImp.drop(['Transported'],axis=1)
colnames_tst = train_encoded_forTst.columns

RobScal = RobustScaler()

train_EncScal_array_forTst = RobScal.fit_transform(train_encoded_forTst)
train_EncScal_forTst = pd.DataFrame(train_EncScal_array_forTst,columns=colnames_tst)

test_EncScal_array = RobScal.transform(test_encoded)
test_EncScal = pd.DataFrame(test_EncScal_array,columns = colnames_tst)

imputer_forTst= KNNImputer(n_neighbors=30).fit(train_EncScal_forTst)

test_EncScalImp_array = imputer_forTst.transform(test_EncScal)
test_EncScalImp = pd.DataFrame(test_EncScalImp_array, columns=colnames_tst)

#---------------------------Quito estandarización y codificación a test-------------------------- 
test_EncImp_array = RobScal.inverse_transform(test_EncScalImp)
test_EncImp = pd.DataFrame(test_EncImp_array,columns=colnames_tst)

test_Imp = test_EncImp.copy()
test_Imp[cat_columns] = enc.inverse_transform(test_EncImp[cat_columns])

#---------------Añado PassengerId a los dos dataframes resultantes---------------------------

train_Imp[['PassengerId']] = passID
test_Imp[['PassengerId']] = passIdtest

#Reordeno columnas
colnames = train_Imp.columns.tolist()
train_Imp = train_Imp[['PassengerId']+colnames[:-1]]

colnames = test_Imp.columns.tolist()
test_Imp = test_Imp[['PassengerId']+colnames[:-1]]

#Pongo como buleanos transported
train_Imp['Transported'] = train_Imp['Transported'].astype('bool')

#--------------------------Exporto en csv los imputados-------------------

train_Imp.to_csv(ROOT + '/data/train_pr_KnnImputed_RobScal.csv',index=False)
test_Imp.to_csv(ROOT + '/data/test_pr_KnnImputed_RobScal.csv',index=False)



