import csv

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import classification_report
from sklearn.model_selection import train_test_split


def pre_process_micro_kol_profile(micro_kol):
    # return features for this micro_kol
    # field 0, field 1, field 2, 3 and label 0 and 1
    return [int(micro_kol[-3]), int(micro_kol[-2])], [int(micro_kol[-1])]


def load_data(csv_file_path):
    # TODO: validate the input file
    X = []
    y = []
    with open(csv_file_path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            item_x, item_y = pre_process_micro_kol_profile(row)
            X.append(item_x)
            y.append(item_y)
    return np.array(X), np.array(y)


# train and validate the logistic regression model
def train_model(X, y):
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr


def test_model(model: LogisticRegression, X, y):
    y_pred = model.predict(X)
    return classification_report(y, y_pred)


# run the prediction
def predict(model: LogisticRegression, X):
    # prediction the probability of label 0 and 1
    return model.predict_log_proba(X)


def main(csv_file_path):
    X, y = load_data(csv_file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr = train_model(X_train, y_train)
    report = test_model(lr, X_test, y_test)
    print(report)
    # save the model
    joblib.dump(lr, 'attractness_lr.pkl')


if __name__ == '__main__':
    # todo: change to a configurable parameter
    main('/sample.csv')
