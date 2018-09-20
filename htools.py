# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Tools needed for hyperbolic geometry:

* minkowski inner product
* conversion to and from poincare ball coordinates

"""
# Standard-library imports
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Third-party imports
import numpy as np
import torch

def mink_prod(x, y):
    """
    (x,y) -> x1*y1 - sum(x2y2, ..., xnyn)
    
    Assume x, y come in batch mode

    """
    if len(x.shape) < 2:
        x = x.reshape(1, -1)
    if len(y.shape) < 2:
        y = y.reshape(1, -1)
    # head = torch.mul(x[:,0], y[:,0])
    # logger.info(head)

    # rest = torch.sum(torch.mul(x, y), 1)

    # return head - rest

    mink_x = x.copy()
    mink_x[:, 1:] = -mink_x[:, 1:]

    return np.sum(mink_x * y, 1).reshape(-1, 1)


def ball2loid(b):
    """
    Convert from poincare ball coordinates to hyperboloid

    """
    x0 = 2. / (1 - np.sum(b**2, 1)) - 1
    x0 = x0.reshape(-1, 1)
    bx0 = b * (x0+1)

    res = np.empty((bx0.shape[0], bx0.shape[1]+1))
    res[:, 0] = x0.ravel()
    res[:, 1:] = bx0

    return res


def loid2ball(l):
    """
    Convert hyperboloid coordinates to poincare ball

    """
    head = l[:,1:]
    rest = 1 + l[:, 0]
    rest = rest.reshape(-1, 1)

    return np.divide(head, rest)


def obj_fn(w, x, y, C):
    if len(y.shape) < 2:
        y = y.reshape(-1, 1)

    margin_term = -mink_prod(w, w)/2.0
    misclass_term = np.arcsinh(1) - np.arcsinh(y * mink_prod(x, w))
    obj = margin_term + C * np.sum(misclass_term)

    return obj.ravel()


def grad_fn(w, x, y, C):
    if len(y.shape) < 2:
        y = y.reshape(-1, 1)

    w_grad_margin = w.copy()
    w_grad_margin[0] = -1. * w_grad_margin[0]
    z = y * mink_prod(x, w)
    missed = (np.arcsinh(1) -np.arcsinh(z)) > 0
    x_grad_misclass = x
    x_grad_misclass[:, 1:] = -1. * x_grad_misclass[:, 1:]

    w_grad_misclass = missed * -(1. / np.sqrt(1 + z**2)) * y  * x_grad_misclass

    w_grad = w_grad_margin + C * np.sum(w_grad_misclass, 0)

    return w_grad


def is_feasible(w):
    """
    Mink prod of weights should be less than 0

    """
    return (mink_prod(w, w) < 1).ravel().item()


def project_weight(w, alpha, ep=1e-5):
    """
    This function can be minimized to find the smallest alpha, which projects
    weights to the closest point so that w * w = -1 (minkowski)

    """
    new_w = w.copy()
    new_w[1:] = (1 + alpha) * new_w[1:]
    new_w[0] = np.sqrt(np.sum((new_w[1:] - ep)**2))

    return new_w
