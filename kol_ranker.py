#!/usr/bin/python
import argparse
import numpy as np
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
    return X, y, group_by_query(query_id)


def group_by_query(query_id):
    group = []
    n_doc = 1
    q_id_cur = query_id[0]
    for id in query_id[1:]:
        if id == q_id_cur:
            n_doc += 1
        else:
            group.append(n_doc)
            n_doc = 1
            q_id_cur = id
    group.append(n_doc)
    return group


# train the ranker
def train_model(X_train, y_train, group_train, X_valid, y_valid, group_valid):
    params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
              'gamma': 1.0, 'min_child_weight': 0.1,
              'max_depth': 6, 'n_estimators': 4}
    model = xgb.sklearn.XGBRanker(**params)
    model.fit(X_train, y_train, group_train, eval_set=[(X_valid, y_valid)], eval_group=[group_valid])
    return model

def p_at_k(k,pred,y,group):
    '''compute the averaged p@k'''
    # each query corresponds to one group
    cur_start=0
    p_sum=0.
    n_query=len(group)
    for num_doc in group:
        cur_pred=pred[cur_start:cur_start+num_doc]
        cur_y=y[cur_start:cur_start+num_doc]
        p_sum +=_p_at_k(k, cur_pred, cur_y)
        cur_start+=num_doc
    return p_sum / float(n_query)


def _p_at_k(k, pred, y):
    index = np.argsort(pred)
    if k>len(index):
        k=len(index)
    top_k_index = index[-1:-(k+1):-1]
    p = np.sum(y[top_k_index]>0) / float(k)
    return p

def test_model(model: xgb.sklearn.XGBRanker, X, y, group):
    pred = model.predict(X)
    # todo: compute p@10
    p_at_10 = p_at_k(10,pred,y,group)
    print('p@10={}'.format( p_at_10))


def predict(model: xgb.sklearn.XGBRanker, X):
    # return a float number for each x in X, which can be used to rank
    # all in one group
    return model.predict(X)


def main(train_data_path: str, valid_data_path: str, model_path: str):
    X_train, y_train, group_train = load_data(train_data_path)
    X_valid, y_valid, group_valid = load_data(valid_data_path)

    # todo: run cross validation
    print('training...')
    ranker = train_model(X_train, y_train, group_train, X_valid, y_valid, group_valid)
    # todo: run test and prediction demo
    test_model(ranker, X_valid, y_valid, group_valid)
    print('save the model')
    joblib.dump(ranker, model_path)
    print('done')


if __name__ == '__main__':
    # example usage: kol_ranker.py --model_path kol_ranker.pkl --train_data_path kol_ranker.train.txt --valid_data_path kol_ranker.valid.txt
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('--valid_data_path', required=True)
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()
    main(train_data_path=args.train_data_path, valid_data_path=args.valid_data_path, model_path=args.model_path)
