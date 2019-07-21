from sklearn.linear_model import LogisticRegression
from   sklearn.model_selection import train_test_split
from sklearn.metrics.classification import classification_report
import numpy as np

# prepare data

def pre_process_micro_kol_profile(micro_kol):
    # return features for this micro_kol
    return None

def load_data():
    return None

# train and validate the logistic regression model
def train_model(X,y):
    lr = LogisticRegression()
    lr.fit(X,y)
    return lr

# save the model

def test_model(model:LogisticRegression, X, y):
    y_pred = model.predict(X)
    return classification_report(y, y_pred)

# run the prediction
def predict(model:LogisticRegression,X):
    # prediction the probability of label 0 and 1
    return model.predict_log_proba(X)