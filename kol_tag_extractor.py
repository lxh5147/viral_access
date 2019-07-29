# extract candidate tags and fine tune the candidates using advanced semantic similarity
import argparse
import csv
import re

import nltk

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


def process_document(document, stemmer: WordNetLemmatizer):
    document = re.sub(r'\W', ' ', document)
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
    return document


def load_data(train_data_path, stemmer: WordNetLemmatizer):
    '''load background data to construct the vocabulary and df'''
    documents = []
    with open(train_data_path) as f:
        reader = csv.reader(f)
        for item in reader:
            document = process_document(item[-1], stemmer)
            documents.append(document)

    #todo: fine tune the min df and max df
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=[1, 3], min_df=2, max_df=0.7,
                             stop_words=stopwords.words('english'))
    vectorizer.fit(documents)
    return vectorizer


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

def extract_keywords(documents,number_of_keywords,stemmer:WordNetLemmatizer,vectorizer : TfidfVectorizer):
    raw_documents =[]
    for document in documents:
        raw_documents.append(process_document(document,stemmer))
    tf_idf_vector = vectorizer.transform(raw_documents)
    feature_names = vectorizer.get_feature_names()
    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items, number_of_keywords)
    return keywords

def main(train_data_path: str, model_path: str):
    stemmer = WordNetLemmatizer()
    vectorizer = load_data(train_data_path, stemmer)
    joblib.dump(vectorizer, model_path)
    # run keyword extraction for the extraction
    document='I like china'
    keywords = extract_keywords([document], 2, stemmer,vectorizer)
    print(keywords)

if __name__ == '__main__':
    # example usage: kol_tag_extractor.py --model_path tfidf_model.pkl --train_data_path kol_blogs.csv
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()
    main(train_data_path=args.train_data_path, model_path=args.model_path)
