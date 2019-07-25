import argparse
import csv
import re

import nltk
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.classification import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


def process_each(item, stemmer: WordNetLemmatizer):
    # micro blogs concatenated into one string
    item_x = item[-1]
    # region id
    item_y = int(item[0])
    # process each document
    document = re.sub(r'\W', ' ', item_x)
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    # Lemmatization
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    return document, item_y


def load_data(train_data_path):
    stemmer = WordNetLemmatizer()
    documents = []
    y = []
    with open(train_data_path) as f:
        reader = csv.reader(f)
        for item in reader:
            item_x, item_y = process_each(item, stemmer)
            documents.append(item_x)
            y.append(item_y)

    # tfidf based bag of words model
    vectorizer = CountVectorizer(max_features=15000, ngram_range=[1, 2], min_df=5, max_df=0.7,
                                 stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(documents).toarray()
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    y = np.array(y)
    return X, y


# train and validate the logistic regression model
def train_model(X, y):
    # todo: use explicit arguments or use grid search
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X, y)
    return classifier


def test_model(classifier, X, y):
    y_pred = classifier.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    print(conf_matrix)
    print(report)
    print(accuracy)


# run the prediction
def predict(classifier, X):
    # prediction the region id
    return classifier.predict(X)


def main(train_data_path: str, model_path: str):
    X, y = load_data(train_data_path)
    # todo: run cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    lr = train_model(X_train, y_train)
    test_model(lr, X_test, y_test)
    joblib.dump(lr, model_path)


if __name__ == '__main__':
    # example usage: micro_kol_attract_degree_estimator.py --model_path region.pkl --train_data_path kol_region_sample.csv
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()
    main(train_data_path=args.train_data_path, model_path=args.model_path)
