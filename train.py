#!/usr/bin/env python3
# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Train a simple HSVM model

"""
# Standard-library imports
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Third-party imports
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import make_blobs
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.io
from numba import jit
import torch
import torch.optim as optim
import click
from tqdm import tqdm

# Cross-library imports
from model import HSVM
import htools
import hsvm


def get_random_data():
    """
    Generate data points randomly inside the unit ball

    """
    # X = torch.randn(100, 2).float()
    # Y = torch.randn(100).float()
    # ixs = Y > 0.5
    # Y[ixs] = 1
    # Y[~ixs] = -1
    X = (X - X.mean())/X.std() 
    # normalize points

    pass


def get_ball_data(loid=True):
    """
    Creates blobs from gaussians, standardize to restrict to unit ball, if
    loid's requested convert to hyperboloid

    """
    EPS = 1e-5
    X, Y = make_blobs(
            n_samples=500, centers=2, random_state=42, cluster_std=0.02,
            center_box=(-0.1, 0.1))

    Y[np.where(Y == 0)] = -1

    logger.info('neg cls {}, pos cls {}'.format(sum(Y == -1), sum(Y == 1)))

    X = (X - X.mean())/X.std() 
    # X[np.where(Y == -1)] = X[np.where(Y == 1)] * -1.0
    norms_sq = np.sum(X**2, 1)
    ball_ixs = norms_sq < 1
    logger.info('{} points inside poincare ball'.format(sum(ball_ixs)))

    X, Y = X[ball_ixs], Y[ball_ixs]

    logger.info('neg cls {}, pos cls {}'.format(sum(Y == -1), sum(Y == 1)))

    return X, Y


def get_gaussian_data(path, label_pos=1):
    """Load data from matlab files"""
    data = sp.io.loadmat(path)
    X, Y = data['B'], data['label'].ravel().astype(np.int)

    # X = X_orig[np.where((Y_orig == label1) | (Y_orig == label2))]
    # Y = Y_orig[np.where((Y_orig == label1) | (Y_orig == label2))]

    Y[Y == label_pos] = 1
    Y[Y != label_pos] = -1
    # X = (X - X.mean())/X.std() 

    logger.info('neg cls {}, pos cls {}'.format(sum(Y == -1), sum(Y == 1)))

    # norms = np.sqrt(np.sum(X**2, 1))
    # ball_ixs = norms < 1
    # logger.info('{} points inside poincare ball'.format(sum(ball_ixs)))

    # X, Y = X[ball_ixs], Y[ball_ixs]

    # logger.info('cls 1 {}, cls 2 {}'.format(sum(Y == -1), 
    #                                         sum(Y == 1)))

    return X, Y



def euc_train(X, Y, model, params):
    """
    Train SVM in Euclidean space

    """
    N = len(Y)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    model.train()
    for e in tqdm(range(params['epoch'])):
        perm = torch.randperm(N)
        sum_loss = 0

        for i in tqdm(range(0, N, params['batch_size'])):
            x = X[perm[i:i+params['batch_size']]]
            y = Y[perm[i:i+params['batch_size']]]

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

        logger.info('loss {}'.format(sum_loss))

    model.eval()

    y_true = Y.data.cpu().numpy().ravel()
    preds = model(X.cuda()).data.cpu().numpy().ravel()

    correct = sum((preds * y_true) > 0)
    auc = roc_auc_score(y_true, preds)
    logger.info('acc {} auc {}'.format(correct/N, auc))


def project_weight(w, alpha, ep=1e-5):
    """
    This function can be minimized to find the smallest alpha, which projects
    weights to the closest point so that w * w = -1 (minkowski)

    """
    new_w = w.copy()
    new_w[1:] = (1 + alpha) * new_w[1:]
    new_w[0] = np.sqrt(np.sum((new_w[1:] - ep)**2))

    return new_w



@jit(nopython=True, parallel=True)
def hyper_train(X, Y, params):
    """
    Train SVM in Hyperbolic space. We run manually stochastic gradient descent

    """
    N = len(Y)
    # optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    w = np.random.randn(3, )
    lr = params['lr']
    C = params['c']
    not_feasible_counter = 0

    for e in tqdm(range(params['epoch'])):
        perm = np.arange(N)
        random.shuffle(perm)
        sum_loss = 0

        for i in tqdm(range(0, N, params['batch_size'])):
            x = X[perm[i:i+params['batch_size']]]
            y = Y[perm[i:i+params['batch_size']]]

            grad = htools.grad_fn(w, x, y, C)
            w = w - lr * grad

            if not htools.is_feasible(w):
                # not_feasible_counter += 1
                # logger.info('not feasible ({} times)'.format(not_feasible_counter))
                res = sp.optimize.minimize_scalar(
                    lambda alpha: np.sum((project_weight(w, alpha) - w)**2))
                alpha = res.x
                w = project_weight(w, alpha)

                assert htools.is_feasible(w)

            obj = htools.obj_fn(w, x, y, C)
            
            sum_loss += obj.item()

        logger.info('loss {}'.format(sum_loss))

    
    y_true = Y.ravel()
    preds = htools.mink_prod(X, w).ravel()
    correct = sum((y_true * preds) > 0)
    auc = roc_auc_score(y_true, preds)
    logger.info('acc {} auc {}'.format(correct/N, auc))

    return w


def visualize(X, Y, W, b=None):
    # W = (model.fc.weight[0]).data.cpu().numpy()
    # b = (model.fc.bias[0]).data.cpu().numpy()

    delta = 0.01
    x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    x, y = np.meshgrid(x, y)
    xy = list(map(np.ravel, [x, y]))

    z = None
    if b is None:
        z = (W.dot(xy)).reshape(x.shape)
    else:
        z = (W.dot(xy) + b).reshape(x.shape)

    # z[np.where(z > 1.)] = 4
    z[np.where((z > 0.) & (z <= 1.))] = 3
    z[np.where((z > -1.) & (z <= 0.))] = 2
    # z[np.where(z <= -1.)] = 1

    plt.figure(figsize=(10, 10))
    plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
    plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
    plt.contourf(x, y, z, alpha=0.8, cmap="Greys")
    cls1_ix = np.where(Y == -1)
    cls2_ix = np.where(Y == 1)
    plt.scatter(x=X[cls1_ix, 0], y=X[cls1_ix, 1], marker='x', c="blue", s=15)
    plt.scatter(x=X[cls2_ix, 0], y=X[cls2_ix, 1], marker='o',  c="green", s=10)
    # plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)
    plt.tight_layout()
    plt.show()


def visualize_loid(X_ball, Y, W, b=None):
    """
    Compute function on a 3d grid in hyperboloid coordinates, plot in ball
    coordinates. Input is in ball coordinates

    """
    delta = 0.1
    sample_size = 200

    x_ball_list = np.linspace(X_ball[:, 0].min(), X_ball[:, 0].max(), sample_size)
    y_ball_list = np.linspace(X_ball[:, 1].min(), X_ball[:, 1].max(), sample_size)
    x_ball, y_ball = np.meshgrid(x_ball_list, y_ball_list)
    # xy_ball = list(map(np.ravel, [x_ball, y_ball]))

    f = []

    for i, j in zip(x_ball.ravel(), y_ball.ravel()):
        curr_x = np.array([i, j]).reshape(1, -1)
        curr_x_loid = htools.ball2loid(curr_x)
        value = htools.mink_prod(curr_x_loid, W)
        f.append(value)

    f = np.array(f).reshape(x_ball.shape)

    f[np.where(f > 1.)] = 4
    f[np.where((f > 0.) & (f <= 1.))] = 3
    f[np.where((f > -1.) & (f <= 0.))] = 2
    f[np.where(f <= -1.)] = 1

    plt.figure(figsize=(10, 10))
    plt.xlim([X_ball[:, 0].min() + delta, X_ball[:, 0].max() - delta])
    plt.ylim([X_ball[:, 1].min() + delta, X_ball[:, 1].max() - delta])
    plt.contourf(x_ball, y_ball, f, alpha=0.8, cmap="Greys")
    cls1_ix = np.where(Y == -1)
    cls2_ix = np.where(Y == 1)
    plt.scatter(x=X_ball[cls1_ix, 0], y=X_ball[cls1_ix, 1], marker='x', c="blue", s=15)
    plt.scatter(x=X_ball[cls2_ix, 0], y=X_ball[cls2_ix, 1], marker='o',  c="green", s=10)
    # plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)
    plt.tight_layout()
    plt.show()


def euc_main(X, Y, params):
    model = HSVM(2, mode='euclidean')

    if torch.cuda.is_available():
        model.cuda()

    euc_train(X, Y, model, params)
    visualize(X, Y, (model.fc.weight[0]).data.cpu().numpy(), 
                    (model.fc.bias[0]).data.cpu().numpy())


def hyper_main(X_loid, Y, params):

    w = hyper_train(X_loid, Y, params)
    X_ball = htools.loid2ball(X_loid)
    visualize_loid(X_ball, Y, w)


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--c', type=float, default=0.01)
@click.option('--epoch', type=int, default=100)
@click.option('--lr', type=float, default=0.01)
@click.option('--batch-size', type=int, default=5)
def old_main(path, c, epoch, lr, batch_size):
    params = {
        'c': c,
        'epoch': epoch,
        'lr': lr,
        'batch_size': batch_size,
    }

    # X, Y = get_ball_data()
    X, Y = get_gaussian_data(path)
    X_loid = htools.ball2loid(X)
    logger.info('converting from ball to loid {} -> {}'.format(X.shape, X_loid.shape))
    X_euc = torch.from_numpy(X).float()
    Y_euc = torch.from_numpy(Y).float()
    
    logger.info('euclidean SVM')
    euc_main(X_euc, Y_euc, params)
    logger.info('hyperbolic SVM')
    hyper_main(X_loid, Y, params)



@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--c', type=float, default=1.0)
@click.option('--epochs', type=int, default=100)
@click.option('--lr', type=float, default=0.01)
@click.option('--batch-size', type=int, default=16)
@click.option('--pretrained', is_flag=True, default=False)
@click.option('--label-pos', type=int, default=1, help='Positive label')
def main(path, c, epochs, lr, batch_size, pretrained, label_pos):
    params = {
        'C': c,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'pretrained': pretrained,
    }

    logger.info('params {}'.format(params))

    # X, Y = get_ball_data()
    X, Y = get_gaussian_data(path, label_pos=label_pos)
    X_loid = htools.ball2loid(X)
    logger.info('converting from ball to loid {} -> {}'.format(X.shape, X_loid.shape))
    
    logger.info('logistic regression')
    log_regr = LogisticRegression(max_iter=epochs)
    scores = cross_val_score(log_regr, X, Y, scoring='roc_auc')
    logger.info('ROC AUC: {:.2f} +/- {:.2f} ({})'.format(np.mean(scores), np.std(scores), scores))
    logger.info('euclidean linear SVM')
    euc_SVM = LinearSVC(C=c, max_iter=epochs)
    scores = cross_val_score(euc_SVM, X, Y, scoring='roc_auc')
    logger.info('ROC AUC: {:.2f} +/- {:.2f} ({})'.format(np.mean(scores), np.std(scores), scores))
    euc_SVM.fit(X, Y)
    visualize(X, Y, euc_SVM.coef_.ravel())
    logger.info('hyperbolic linear SVM')
    hyp_SVM = hsvm.LinearHSVM(**params)
    scores = cross_val_score(hyp_SVM, X_loid, Y, scoring='roc_auc')
    logger.info('ROC AUC: {:.2f} +/- {:.2f} ({})'.format(np.mean(scores), np.std(scores), scores))
    hyp_SVM.fit(X_loid, Y)
    visualize_loid(X, Y, hyp_SVM.coef_.ravel())


if __name__ == '__main__':
    main()
