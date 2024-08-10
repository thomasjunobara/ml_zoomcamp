#!/usr/bin/env python
# coding: utf-8

# # 5 Deploying Machine Learning Models
# 
# - we'll use the same model we trained and evaluated previously - the churn prediction model. Now we'll deploy it as a web service.

# # 5.2 Saving and loading the model

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'


# Parameters
C = 1.0
n_splits = 5
output_file = f'model_C{C}.bin'


# Data preparation

df = pd.read_csv(data)

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

# Split the df into training, validation, test

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = ['gender', 
               'seniorcitizen', 
               'partner', 
               'dependents', 
               'phoneservice', 
               'multiplelines', 
               'internetservice', 
               'onlinesecurity', 
               'onlinebackup', 
               'deviceprotection', 
               'techsupport',
               'streamingtv', 
               'streamingmovies', 
               'contract', 
               'paperlessbilling',
               'paymentmethod']

def train(df_train, y_train, C=1.0): # the parameter C is used to tune our model (default vale = 1)
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# Train our final model


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc

# The score is a little bit better than 0.841

# Save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)