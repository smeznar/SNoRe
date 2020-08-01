import argparse
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import defaultdict
from six import iteritems

from src.snore import SNoRe
from src.utils import from_mat_file, TopKRanker


def evaluate_snore(network, labels, num_shuffles=10, all=False, data_perc=0.5):
    # Embedding
    model = SNoRe()
    emb = model.embed(network)

    # Evaluation
    all_results = defaultdict(list)
    shuffles = []
    for x in range(num_shuffles):
        shuffles.append(skshuffle(emb, labels))
        if all:
            training_percents = np.asarray(range(1, 10)) * .1
        else:
            training_percents = [data_perc]
        for train_percent in training_percents:
            for shuf in shuffles:
                X, y = shuf

                training_size = int(train_percent * X.shape[0])

                X_train = X[:training_size, :]
                y_train_ = y[:training_size]
                y_train = [list() for x in range(y_train_.shape[0])]

                cy = y_train_.tocoo()
                for i, j in zip(cy.row, cy.col):
                    y_train[i].append(j)

                assert sum(len(l) for l in y_train) == y_train_.nnz

                X_test = X[training_size:, :]
                y_test_ = y[training_size:]

                y_test = [[] for _ in range(y_test_.shape[0])]

                cy = y_test_.tocoo()
                for i, j in zip(cy.row, cy.col):
                    y_test[i].append(j)

                clf = TopKRanker(LogisticRegression())
                clf.fit(X_train, y_train_)

                # find out how many labels should be predicted
                top_k_list = [len(l) for l in y_test]
                preds = clf.predict(X_test, top_k_list)

                results = {}
                averages = ["micro", "macro"]
                for average in averages:
                    results[average] = f1_score(mlb.fit_transform(y_test),
                                                    mlb.fit_transform(preds), average=average)

                    all_results[train_percent].append(results)
    print('Results of SNoRe using embeddings of dimensionality', emb.shape[1])
    print('-------------------')
    for train_percent in sorted(all_results.keys()):
        print('Train percent:', train_percent)
        avg_score = defaultdict(float)
        for score_dict in all_results[train_percent]:
            for metric, score in iteritems(score_dict):
                avg_score[metric] += score
        for metric in avg_score:
            avg_score[metric] /= len(all_results[train_percent])
        print('Average score:', dict(avg_score))
        print('-------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Embedding Evaluation Utility')
    parser.add_argument('--dataset', help='Path to the dataset',
                        required=True, action='store')
    parser.add_argument("--num_shuffles", default=10, type=int, help='Number of shuffles.')
    parser.add_argument("--all", default=False, action='store_true',
                        help='The embeddings are evaluated on all training percents from 10 to 90'
                             ' when this flag is set to true.')
    parser.add_argument("--data_perc", default=0.5, type=float, help='Data percentage if all is not chosen.')
    # parser.add_argument("--out_file", default="embedding.npz", help='Filename where embedding is saved if flag'
    #                                                                 '--save is added')
    # parser.add_argument("--save")
    args = parser.parse_args()

    network, labels, mlb = from_mat_file(args.dataset)
    evaluate_snore(network, labels, args.num_shuffles, args.all, args.data_perc)
