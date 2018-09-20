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
import numpy as np
import scipy as sp
import scipy.optimize
from sklearn.metrics import roc_auc_score
import torch
import torch.optim as optim

# Cross-library imports
from model import HSVM
import htools


class KGEvaluator(object):
    """Collects embedding files, associates them with the matching pairs of
    train/test files for single-label classification task, and evaluates these
    embeddings with a logistic regression"""
    def __init__(self):
        # representation functions for one arc from two embeddings of a node
        self.repr_fns = {
            "sum": lambda x, y: x + y,
            "mean": lambda x, y: (x + y) / 2,
            "mult": lambda x, y: x * y,
            "concat": lambda x, y: np.concatenate([x, y]),
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


    def convert_data(self, f, E, binary_op="concat"):
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
            X = np.zeros((N, d)) 
            y = np.ones(N).astype(np.int)

            # collect how many missing embeddings we get
            missing = []
            # read in v1, v2, labels and compute X, y
            for i, l in enumerate(lines):
                v1, v2, label = l.strip().split()

                # if no embedding found that delete this example
                try:
                    emb = repr_fn(E[v1], E[v2])

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
                        binary_op='concat', classifier_type='euc_svm'):
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

            data = (X_tr, X_te, y_tr, y_te)

            logger.info('euclidean SVM')
            self.eval_euc_svm(data, params)

            logger.info('hyperbolic SVM')
            self.eval_hyper_svm(data, params)


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


    def eval_euc_svm(self, data, params):
        """Evaluate euclidean SVM on given data"""
        X_tr, X_te, Y_tr, Y_te = [torch.from_numpy(x) for x in data]
        X_tr, X_te = X_tr.float(), X_te.float()
        Y_tr, Y_te = Y_tr.float(), Y_te.float()
        Y_tr[Y_tr == 0] = -1.0
        Y_te[Y_te == 0] = -1.0
        model = HSVM(10, mode='euclidean')

        if torch.cuda.is_available():
            model.cuda()

        N = len(Y_tr)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])

        model.train()
        for e in tqdm(range(params['epoch'])):
            perm = torch.randperm(N)
            sum_loss = 0
            acc_i = 0.0

            for i in tqdm(range(0, N, params['batch_size'])):
                x = X_tr[perm[i:i+params['batch_size']]]
                y = Y_tr[perm[i:i+params['batch_size']]]

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                optimizer.zero_grad()
                output = model(x)

                loss = torch.mean(torch.clamp(1 - output * y, min=0))
                loss += params['c'] * torch.mean(model.fc.weight ** 2)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()

                y_true_i = y.cpu().numpy().ravel()
                preds_i = output.data.cpu().numpy().ravel()

                correct_i = sum((preds_i * y_true_i) > 0)
                acc_i += correct_i

            logger.info('train loss {}, acc {}'.format(sum_loss, acc_i/len(Y_tr)))

        model.eval()

        y_true = Y_te.data.cpu().numpy().ravel()
        preds = model(X_te.cuda()).data.cpu().numpy().ravel()

        correct = sum((preds * y_true) > 0)
        auc = roc_auc_score(y_true, preds)
        logger.info('test: acc {} auc {}'.format(correct/len(Y_te), auc))



    def eval_hyper_svm(self, data, params):
        """
        Train SVM in Hyperbolic space. We run manually stochastic gradient descent

        """
        X_tr, X_te, Y_tr, Y_te = data
        Y_tr[Y_tr == 0] = -1.0
        Y_te[Y_te == 0] = -1.0

        N = len(Y_tr)
        w = np.random.randn(10, )
        lr = params['lr']
        C = params['c']
        not_feasible_counter = 0

        for e in tqdm(range(params['epoch'])):
            perm = np.arange(N)
            random.shuffle(perm)
            sum_loss = 0
            acc_i = 0.0

            for i in tqdm(range(0, N, params['batch_size'])):
                x = X_tr[perm[i:i+params['batch_size']]]
                y = Y_tr[perm[i:i+params['batch_size']]]

                grad = htools.grad_fn(w, x, y, C)
                w = w - lr * grad

                if not htools.is_feasible(w):
                    # not_feasible_counter += 1
                    # logger.info('not feasible ({} times)'.format(not_feasible_counter))
                    res = sp.optimize.minimize_scalar(
                        lambda alpha: np.sum((htools.project_weight(w, alpha) - w)**2))
                    alpha = res.x
                    w = htools.project_weight(w, alpha)

                    assert htools.is_feasible(w)

                obj = htools.obj_fn(w, x, y, C)
                
                sum_loss += obj.item()

            y_true_tr = Y_tr.ravel()
            preds_tr = htools.mink_prod(X_tr, w).ravel()

            correct_tr = sum((preds_tr * y_true_tr) > 0)
            logger.info('train loss {}, acc {}'.format(sum_loss, correct_tr/len(Y_tr)))

        
        y_true = Y_te.ravel()
        preds = htools.mink_prod(X_te, w).ravel()
        correct = sum((y_true * preds) > 0)
        auc = roc_auc_score(y_true, preds)
        logger.info('acc {} auc {}'.format(correct/len(Y_te), auc))


@click.command()
@click.argument('emb_path', type=click.Path(exists=True))
@click.argument('tr_path', type=click.Path(exists=True))
@click.argument('te_path', type=click.Path(exists=True))
@click.option('--c', type=float, default=0.01)
@click.option('--epoch', type=int, default=100)
@click.option('--lr', type=float, default=0.01)
@click.option('--batch-size', type=int, default=32)
def main(emb_path, tr_path, te_path, c, epoch, lr, batch_size):
    """
    Args:
        emb_path - path to embedding file

    """
    params = {
        'c': c,
        'epoch': epoch,
        'lr': lr,
        'batch_size': batch_size,
    }
    my_evaluator = KGEvaluator()
    my_evaluator.evaluate_params(emb_path, tr_path, te_path, params)


if __name__ == '__main__':
    main()
