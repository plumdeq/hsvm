# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Simple pytorch model for Hyperboloid SVM

"""
import torch
import torch.nn as nn


class HSVM(nn.Module):
    """Hyperboloid lives in n+1 dimensions"""
    def __init__(self):
        super(HSVM, self).__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        h = self.fc(x)
        return h
