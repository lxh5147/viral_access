import argparse
import csv

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import classification_report
from sklearn.model_selection import train_test_split


def process_each(item):
    # return features for this micro_kol
    # field 0, field 1, field 2, 3 and label 0 and 1
    # todo: further data normalization
    return [int(item[-3]), int(item[-2])], [int(item[-1])]


def load_data(train_data_path):
    # TODO: validate the input file
    X = []
    y = []
    with open(train_data_path, 'rb') as f:
        reader = csv.reader(f)
        for item in reader:
            item_x, item_y = process_each(item)
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


def main(train_data_path: str, model_path: str):
    X, y = load_data(train_data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr = train_model(X_train, y_train)
    report = test_model(lr, X_test, y_test)
    print(report)
    joblib.dump(lr, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()
    main(train_data_path=args.train_data_path, model_path=args.model_path)
