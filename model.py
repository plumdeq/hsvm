# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Simple pytorch model for Hyperboloid SVM

"""
import torch
import torch.nn as nn


class HSVM(nn.Module):
    """Hyperboloid lives in n+1 dimensions"""
    def __init__(self, n, mode='hyperbolic'):
        super(HSVM, self).__init__()
        if mode == 'hyperbolic':
            n = n + 1

        self.fc = nn.Linear(n, 1)

    def forward(self, x):
        h = self.fc(x)
        return h
