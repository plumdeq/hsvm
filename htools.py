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
import scipy as sp
import torch
from numba import jit


# =================
# HYPERBOLOID MODEL
# =================


@jit(parallel=True)
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
    xy = np.float64(mink_x) * np.float64(y)

    return np.sum(xy, 1).reshape(-1, 1)


def ball2loid(b):
    """
    Convert from poincare ball coordinates to hyperboloid

    """
    x0 = 2. / (1 - np.sum(b**2, 1)) - 1
    x0 = x0.reshape(-1, 1)
    bx0 = b * (x0+1)

    res = np.empty((bx0.shape[0], bx0.shape[1]+1), dtype=np.float64)
    res[:, 0] = x0.ravel()
    res[:, 1:] = bx0

    return res


def loid_dist(x, y, from_ball=True):
    """
    Hyperbolic distance between x and y. We compute it in the hyperboloid
    model, thus convert to loid from ball, if coordinates are in ball

    """
    if from_ball:
        # should be in batch x dims 
        x = ball2loid(x.reshape(1, -1))
        y = ball2loid(y.reshape(1, -1))

    return np.arccosh(mink_prod(x, y).ravel().item())


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


@jit(parallel=True)
def grad_fn(w, x, y, C):
    if len(y.shape) < 2:
        y = y.reshape(-1, 1)

    w_grad_margin = w.copy()
    w_grad_margin[0] = -1. * w_grad_margin[0]
    z = np.float64(y * mink_prod(np.float64(x), np.float64(w)))
    missed = (np.arcsinh(1) -np.arcsinh(z)) > 0
    x_grad_misclass = x
    x_grad_misclass[:, 1:] = -1. * x_grad_misclass[:, 1:]

    sqrt_term = np.float64(1.0 + z**2)
    w_grad_misclass = missed * -(1. / np.sqrt(sqrt_term)) * y  * x_grad_misclass

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


# =============
# POINCARE BALL
# =============

def mobius_addition(x, y, c=1.0):
    """
    Mobius addition in the poincare ball model

    """
    xx = np.sum(x*x)
    yy = np.sum(y*y)
    xy = np.sum(x*y)

    numerator = (1 + 2*c*xy + c*yy)*x + (1 - c*xx)*y
    denominator = 1 + 2*c*xy + c*c*xx*yy

    return numerator/denominator


def project_to_unitball(x, eps=1e-5):
    """
    If norm of x > 1, then you need to bring it back to the poincare ball. If
    `optimize=False` then project recursively

    """
    if np.sum(x*x) < 1:
        return x

    x_norm = np.sqrt(np.sum(x*x))
    x = x/x_norm - eps

    return x


def ball_dist(x, y):
    xx = np.sum(x*x)
    yy = np.sum(y*y)
    x_minus_y = x - y
    x_minus_y2 = np.sum(x_minus_y*x_minus_y)

    arg = 2*(x_minus_y2)/((1 - xx)*(1 - yy))

    return np.arccosh(1 + arg)


def poincare_metric(x):
    """
    Poincare metric of x, i.e., inner product of x. x shape should be  1 x d

    """
    x_new = x.copy().reshape(1, -1)
    euc_inner = np.sum(x_new**2)
    conformal_factor = 2 / (1 - euc_inner)

    return conformal_factor**2 * euc_inner
