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
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import torch
import torch.optim as optim
import click
from tqdm import tqdm

# Cross-library imports
from model import HSVM
import htools


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

    logger.info('cls 1 {}, cls 2 {}'.format(sum(Y == -1), 
                                            sum(Y == 1)))

    X = (X - X.mean())/X.std() 
    X[np.where(Y == -1)] = X[np.where(Y == 1)] * -1.0
    norms = np.sqrt(np.sum(X**2, 1))
    ball_ixs = norms < 1
    logger.info('{} points inside poincare ball'.format(sum(ball_ixs)))

    X, Y = X[ball_ixs], Y[ball_ixs]

    logger.info('cls 1 {}, cls 2 {}'.format(sum(Y == -1), 
                                            sum(Y == 1)))

    if loid:
        prev_X = X
        X = htools.ball2loid(X)
        logger.info('converting from ball to loid {} -> {}'.format(prev_X.shape, X.shape))
    
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



def hyper_train(X, Y, params):
    """
    Train SVM in Hyperbolic space. We run manually stochastic gradient descent

    """
    N = len(Y)
    # optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    w = np.random.randn(3, )
    lr = params['lr']
    C = params['c']

    for e in tqdm(range(params['epoch'])):
        perm = np.arange(N)
        random.shuffle(perm)
        sum_loss = 0

        for i in tqdm(range(0, N, params['batch_size'])):
            x = X[perm[i:i+params['batch_size']]]
            y = Y[perm[i:i+params['batch_size']]]

            grad = htools.grad_fn(w, x, y, C)
            w = w - lr * grad
            obj = htools.obj_fn(w, x, y, C)
            
            sum_loss += obj.item()

        logger.info('loss {}'.format(sum_loss))


def visualize(X, Y, model):
    W = (model.fc.weight[0]).data.cpu().numpy()
    b = (model.fc.bias[0]).data.cpu().numpy()

    delta = 0.01
    x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    x, y = np.meshgrid(x, y)
    xy = list(map(np.ravel, [x, y]))

    z = (W.dot(xy) + b).reshape(x.shape)
    z[np.where(z > 1.)] = 4
    z[np.where((z > 0.) & (z <= 1.))] = 3
    z[np.where((z > -1.) & (z <= 0.))] = 2
    z[np.where(z <= -1.)] = 1

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


@click.command()
@click.option('--c', type=float, default=0.01)
@click.option('--epoch', type=int, default=100)
@click.option('--lr', type=float, default=0.01)
@click.option('--batch-size', type=int, default=5)
def euc_main(c, epoch, lr, batch_size):
    params = {
        'c': c,
        'epoch': epoch,
        'lr': lr,
        'batch_size': batch_size,
    }

    X, Y = get_data()

    model = HSVM(2, mode='euclidean')

    if torch.cuda.is_available():
        model.cuda()

    euc_train(X, Y, model, params)
    visualize(X, Y, model)


@click.command()
@click.option('--c', type=float, default=0.01)
@click.option('--epoch', type=int, default=100)
@click.option('--lr', type=float, default=0.01)
@click.option('--batch-size', type=int, default=5)
def hyper_main(c, epoch, lr, batch_size):
    params = {
        'c': c,
        'epoch': epoch,
        'lr': lr,
        'batch_size': batch_size,
    }

    X, Y = get_ball_data()

    hyper_train(X, Y, params)


if __name__ == '__main__':
    hyper_main()
