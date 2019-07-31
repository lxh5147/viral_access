#!/usr/bin/python
import argparse

import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib


def convert_to_feature_vector(query, kol):
    '''to extract features from a query-document pair'''
    pass


def load_data(train_data_path):
    '''convert to X, y and query id'''
    # 0 qid:18219 1:0.027615 2:0.500000 3:0.750000 4:0.333333 … 45:0.010291 46:0.046400 #docid = GX004-93-7097963 inc = 0.0428115405134536 prob = 0.860366
    X, y, query_id = load_svmlight_file(train_data_path, query_id=True)
    return group_by_query(X, y, query_id)


def group_by_query(X, y, query_id):
    query_group = {}
    for i, id in enumerate(query_id):
        if id in query_group:
            query_group[id].append(id)
        else:
            query_group[id] = [i]
    group = []
    X_ = []
    y_ = []
    for _, g in query_group.items():
        group.append(len(g))
        for i in g:
            X_.append(X[i])
            y_.append(y[i])
    return X_, y_, group


# train the ranker
def train_model(X_train, y_train, group_train, X_valid, y_valid, group_valid):
    params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
              'gamma': 1.0, 'min_child_weight': 0.1,
              'max_depth': 6, 'n_estimators': 4}
    model = xgb.sklearn.XGBRanker(**params)
    model.fit(X_train, y_train, group_train, eval_set=[(X_valid, y_valid)], eval_group=[group_valid])
    return model


def test_model(model: xgb.sklearn.XGBRanker, X, y, group):
    pred = model.predict(X)
    # todo: compute p@10 and ndcg@10
    pass


def predict(model: xgb.sklearn.XGBRanker, X):
    # return a float number for each x in X, which can be used to rank
    return model.predict(X)


def main(train_data_path: str, valid_data_path: str, model_path: str):
    X_train, y_train, group_train = load_data(train_data_path)
    X_valid, y_valid, group_valid = load_data(valid_data_path)

    # todo: run cross validation
    ranker = train_model(X_train, y_train, group_train, X_valid, y_valid, group_valid)
    # todo: run test and prediction demo
    joblib.dump(ranker, model_path)


if __name__ == '__main__':
    # example usage: kol_ranker.py --model_path kol_ranker.pkl --train_data_path kol_ranker.train.txt --valid_data_path kol_ranker.valid.txt
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()
    main(train_data_path=args.train_data_path, model_path=args.model_path)
