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
import torch

def mink_prod(x, y):
    """
    (x,y) -> x1*y1 - sum(x2y2, ..., xnyn)
    
    Assume x, y come in batch mode

    """
    head = torch.mul(x[:,0], y[:,0])
    logger.info(head)

    rest = torch.sum(torch.mul(x, y), 1)

    return head - rest


def ball2loid(b):
    """
    Convert from poincare ball coordinates to hyperboloid

    """
    x0 = 2. / (1 - torch.sum(torch.mul(b, b), 1)) - 1
    x0 = x0.view(-1, 1)
    bx0 = torch.mul(b, (x0+1))

    return torch.cat((x0, bx0), dim=1)


def loid2ball(l):
    """
    Convert hyperboloid coordinates to poincare ball

    """
    head = l[:,1:]
    rest = 1 + l[:, 0]
    rest = rest.view(-1, 1)

    return torch.div(head, rest)
