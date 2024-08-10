# ### Load the model
# - restart the kernel before

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path

model_file = Path(__file__).parent / 'model_C1.0.bin'

with open(model_file, 'rb') as f_in: # if we don't change the wb to rb, it will overwrite the file
    (dv, model) = pickle.load(f_in)


customer = {
    'gender': 'female', 
    'seniorcitizen': 0, 
    'partner': 'yes', 
    'dependents': 'no', 
    'phoneservice': 'no', 
    'multiplelines': 'no_phone_service', 
    'internetservice': 'dsl', 
    'onlinesecurity': 'no', 
    'onlinebackup': 'yes', 
    'deviceprotection': 'no', 
    'techsupport': 'no',
    'streamingtv': 'no', 
    'streamingmovies': 'no', 
    'contract': 'month-to-month', 
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85,
}


X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]

print('input', customer)
print('churn probability', y_pred)