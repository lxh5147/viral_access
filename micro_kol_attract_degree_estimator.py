import argparse
import csv

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


def to_int(s):
    try:
        return int(s)
    except ValueError:
        return int(float(s))


def process_each(item):
    # return features for this micro_kol
    # field 0, field 1, field 2, 3 and label 0 and 1
    # todo: further data normalization
    item_x = [to_int(item[-5]), to_int(item[-4]), to_int(item[-3]), to_int(item[-2])]
    # item_y can be 0, 1
    item_y = to_int(item[-1])
    return item_x, item_y


def load_data(train_data_path):
    # TODO: validate the input file
    X = []
    y = []
    with open(train_data_path) as f:
        reader = csv.reader(f)
        for item in reader:
            item_x, item_y = process_each(item)
            X.append(item_x)
            y.append(item_y)
    return np.array(X), np.array(y)


# train and validate the logistic regression model
def train_model(X, y):
    # todo: use explicit arguments or use grid search
    # GridSearchCV
    classifier = LogisticRegression()
    classifier.fit(X, y)
    return classifier


def test_model(classifier, X, y):
    y_pred = classifier.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)
    accuracy=accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    print( conf_matrix)
    print(report)
    print(accuracy)

# run the prediction
def predict(classifier, X):
    # prediction the probability of label 0 and 1
    return classifier.predict_log_proba(X)


def main(train_data_path: str, model_path: str):
    X, y = load_data(train_data_path)
    # todo: run cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    lr = train_model(X_train, y_train)
    test_model(lr, X_test, y_test)
    joblib.dump(lr, model_path)


if __name__ == '__main__':
    # example usage: micro_kol_attract_degree_estimator.py --model_path test.pkl --train_data_path sample.csv
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()
    main(train_data_path=args.train_data_path, model_path=args.model_path)
