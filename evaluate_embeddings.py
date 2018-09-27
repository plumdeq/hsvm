#!/usr/bin/env python3
# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Input are three files:

1) Embeddings (without header)
2) Train file 'id1 id2 1|0'
3) Test file, same format as 2)

Read in embeddings, train EucSVM and HSVM, compare evaluation results

"""
# Standard-library imports
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Third-party imports
import click
from tqdm import tqdm
#
import numpy as np
import scipy as sp
import scipy.optimize
#
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
#
import torch
import torch.optim as optim

# Cross-library imports
from model import HSVM
import htools
import hsvm
import train


class KGEvaluator(object):
    """Collects embedding files, associates them with the matching pairs of
    train/test files for single-label classification task, and evaluates these
    embeddings with a logistic regression"""
    def __init__(self):
        # representation functions for one arc from two embeddings of a node
        self.repr_fns = {
            "sum": lambda x, y: x + y,
            "mean": lambda x, y: (x + y) / 2.0,
            "mult": lambda x, y: x * y,
            "concat": lambda x, y: np.concatenate([x, y]),
            "diff": lambda x, y: x - y,
            "mobius_sum": lambda x, y: htools.mobius_addition(x, y),
            "mobius_diff": lambda x, y: htools.mobius_addition(x, -y),
            "mobius_sum_mean": lambda x, y: htools.mobius_addition(x, y) / 2.0,
            "mobius_diff_mean": lambda x, y: htools.mobius_addition(x, -y) / 2.0,
        }


    def get_embeddings(self, emb_file):
        # build dictionary "idx: x1 x2"
        logger.info("reading embeddings from {}".format(emb_file))
        E = {}

        with open(emb_file, "r") as f:
            for l in f.readlines():
                line = l.strip().split() 
                E[line[0]] = np.array(list(map(float, line[1:-1]))).astype(np.float32)

        return E


    def convert_data(self, f, E, binary_op="mobius_diff"):
        """read in `v1 v2 label` and train on <E(v1), E(v2)> = 1|0"""

        if binary_op not in self.repr_fns:
            logger.info("{} - unknown representation, falling back to concatenation".format(binary_op))

        repr_fn = self.repr_fns.get(binary_op, self.repr_fns["concat"])
        size = len(E[list(E.keys())[0]])
        
        with open(f, "r") as f:
            lines = f.readlines()

            N = len(lines)

            # if no lines in the file raise exception
            if not N > 0:
                raise Exception("file {} is empty".format(f))

            # take a sample to determine the length of the embeddings
            v1, v2, label = lines[0].strip().split()

            # adjust array size to the representation type
            d = size if binary_op != "concat" else size*2
            X = np.zeros((N, d), dtype=np.float64) 
            y = np.ones(N).astype(np.int)

            # collect how many missing embeddings we get
            missing = []
            # read in v1, v2, labels and compute X, y
            for i, l in enumerate(lines):
                v1, v2, label = l.strip().split()

                # if no embedding found that delete this example
                try:
                    # project back to unit ball
                    emb_v1 = htools.project_to_unitball(E[v1]).astype(np.float64)
                    emb_v2 = htools.project_to_unitball(E[v2]).astype(np.float64)
                    emb = repr_fn(emb_v1, emb_v2).astype(np.float64)

                    X[i] = emb
                    y[i] = int(label)

                except Exception as e:
                    missing.append(i)

        # delete examples with missing embeddings
        if len(missing) > 0:
            X = np.delete(X, missing, 0)
            y = np.delete(y, missing, 0)

        assert i == N-1, "i={}, N={}".format(i, N)

        missing_ratio = len(missing)/N*100
        logger.info("missing embeddings {0}/{1} {2:0.3f}%".format(len(missing), N, missing_ratio))
        logger.info("X shape {}, y shape {}".format(X.shape, y.shape))

        return X, y, missing_ratio


    def evaluate_params(self, emb_file, train_file, test_file, params,
                        binary_op='mobius_diff', classifier_type='euc_svm',
                        visualize=True):
        """
        Evaluate takes a callable for file names, hyperparameter dictionary, the
        representation type of the embedding, and the classifier type. It will then
        decide which hyperparameters to pass to which classifier etc.

        """
        scores = None
        tr_miss_ratio, te_miss_ratio = np.nan, np.nan
        E = self.get_embeddings(emb_file)

        try:
            X_tr, y_tr, tr_miss_ratio = self.convert_data(train_file, E, binary_op=binary_op) 
            X_te, y_te, te_miss_ratio = self.convert_data(test_file, E, binary_op=binary_op) 

            logger.info('prepare data for euclidean SVM')
            data = (X_tr, X_te, y_tr, y_te)
            logger.info('prepare data for hyperbolic SVM')
            data_loid = (htools.ball2loid(X_tr), htools.ball2loid(X_te), y_tr, y_te)

            logger.info('euclidean SVM')
            self.eval_euc_svm(data, params, visualize=visualize)

            logger.info('hyperbolic SVM')
            self.eval_hyper_svm(data_loid, params, visualize=visualize)


            # elif classifier_type == "mlp": 
            #     if "hidden_layer_sizes" in params:
            #         scores = self.evaluate_mlp(data, params["hidden_layer_sizes"])
            #     else:
            #         scores = self.evaluate_mlp(data)
            # else:
            #     raise Exception("Unknown classifier type {}".format(classifier_type))
        except Exception as e:
            logger.info("Could not evaluate")
            logger.info(e)
            scores = {
                "Mean acc": np.nan,
                "F-measure": np.nan,
                "ROC AUC": np.nan,
            }


        # logger.info(",".join(["{}: {}".format(k,v) for k, v in params.items()]))

        # for metric, score in scores.items():
        #     logger.info("{0} on test data {1:0.3f}".format(metric, score))

        # return {
        #     **scores,
        #     **params,
        #     "tr_miss_ratio": tr_miss_ratio,
        #     "te_miss_ratio": te_miss_ratio,
        # }


    def eval_euc_svm(self, data, params, visualize=True):
        """Evaluate euclidean SVM on given data"""
        X_tr, X_te, Y_tr, Y_te = data
        Y_tr[Y_tr == 0] = -1.0
        Y_te[Y_te == 0] = -1.0

        # logger.info('CV on train data')
        # euc_SVM = LinearSVC(C=params['C'], max_iter=params['epochs'])
        # scores = cross_val_score(euc_SVM, X_tr, Y_tr, scoring='roc_auc')
        # logger.info('Train ROC AUC: {:.2f} +/- {:.2f} ({})'.format(np.mean(scores), np.std(scores), scores))

        # euc_SVM = LinearSVC(C=params['C'], max_iter=params['epochs'])
        # euc_SVM.fit(X_tr, Y_tr)
        # te_score = euc_SVM.score(X_te, Y_te)
        # te_auc = roc_auc_score(Y_te, euc_SVM.decision_function(X_te))
        # logger.info('test accuracy {}, ROC AUC {}'.format(te_score, te_auc))

        logger.info('(euc svm) grid search hyperparameter tunning')
        param_grid = { 'C': [0.1, 1, 10] }
        clf = GridSearchCV(estimator=LinearSVC(max_iter=params['epochs']), 
                           param_grid=param_grid, 
                           scoring='roc_auc', n_jobs=-1)
        clf.fit(X_tr, Y_tr)
        logger.info('(train) best roc auc: {:.3f}, best params_ {}'.format(
            clf.best_score_, clf.best_params_))

        roc_auc_te = clf.score(X_te, Y_te)
        roc_auc_te2 = roc_auc_score(Y_te, clf.decision_function(X_te))
        logger.info('(test) roc auc: {:.3f} ({:.3f})'.format(roc_auc_te, roc_auc_te2))

        if visualize:
            train.visualize(X_te, Y_te, clf.best_estimator_.coef_.ravel())


    def eval_hyper_svm(self, data, params, visualize=True):
        """
        Train SVM in Hyperbolic space. We run manually stochastic gradient descent

        """
        X_tr, X_te, Y_tr, Y_te = data
        Y_tr[Y_tr == 0] = -1.0
        Y_te[Y_te == 0] = -1.0

        # logger.info('CV on train data')
        # hyp_SVM = hsvm.LinearHSVM(**params)
        # scores = cross_val_score(hyp_SVM, X_tr, Y_tr, scoring='roc_auc')
        # logger.info('Train ROC AUC: {:.2f} +/- {:.2f} ({})'.format(np.mean(scores), np.std(scores), scores))

        # hyp_SVM = hsvm.LinearHSVM(**params)
        # hyp_SVM.fit(X_tr, Y_tr)
        # te_score = hyp_SVM.score(X_te, Y_te)
        # te_auc = roc_auc_score(Y_te, hyp_SVM.decision_function(X_te))
        # logger.info('test accuracy {}, ROC AUC {}'.format(te_score, te_auc))

        logger.info('(hyp svm) grid search hyperparameter tunning')
        param_grid = { 
            'C': [0.1, 1, 10], 
            'batch_size': [16, 32, 64], 
            'epochs': [100, 500],
            'lr': [0.1, 0.01, 0.001],
            'pretrained': [False, True],
            }
        clf = GridSearchCV(estimator=hsvm.LinearHSVM(**params), param_grid=param_grid, 
                           scoring='roc_auc', n_jobs=-1)
        clf.fit(X_tr, Y_tr)
        logger.info('(train) best roc auc: {:.3f}, best params_ {}'.format(
            clf.best_score_, clf.best_params_))

        roc_auc_te = clf.score(X_te, Y_te)
        logger.info('(test) roc auc: {:.3f}'.format(roc_auc_te))

        if visualize:
            X_te_ball = htools.loid2ball(X_te)
            train.visualize_loid(X_te_ball, Y_te, clf.best_estimator_.coef_.ravel())


@click.command()
@click.argument('emb_path', type=click.Path(exists=True))
@click.argument('tr_path', type=click.Path(exists=True))
@click.argument('te_path', type=click.Path(exists=True))
@click.option('--c', type=float, default=1.0)
@click.option('--epochs', type=int, default=100)
@click.option('--lr', type=float, default=0.01)
@click.option('--batch-size', type=int, default=32)
@click.option('--pretrained', is_flag=True, default=False)
@click.option("--binary-op", default='concat', type=click.Choice(["concat", "mean", "sum", "mult", "diff", "mobius_sum", "mobius_diff", "mobius_sum_mean", "mobius_diff_mean"]))
@click.option("--visualize", is_flag=True, default=False)
def main(emb_path, tr_path, te_path, c, epochs, lr, batch_size, pretrained,
         binary_op, visualize):
    """
    Args:
        emb_path - path to embedding file

    """
    params = {
        'C': c,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'pretrained': pretrained,
    }
    my_evaluator = KGEvaluator()
    my_evaluator.evaluate_params(emb_path, tr_path, te_path, params, 
                                 binary_op=binary_op, visualize=visualize)


if __name__ == '__main__':
    main()
