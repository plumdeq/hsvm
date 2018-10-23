#!/usr/bin/env python3
# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Config file for the evaluation of HSVM vs. EucSVM on bio-kg

"""
# Standard-library imports
import os
from glob import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PATH = "/media/Warehouse/nips2018-rel_learning/umls"
EXCLUDE_DIRS = ['space-separated']
DIM = 10
EPOCHS = 500

EMB_GLOB = lambda fold, dim, epochs: 'global-train-fold-{}-dimension-{}*epochs-{}'.format(fold, dim, epochs)
TR_GLOB = lambda fold: '*train_biased*fold-{}'.format(fold)
TE_GLOB = lambda fold: '*test_biased*fold-{}'.format(fold)

ratios = [80]
folds = list(range(1, 11))

confs = {}


def collect_subfolders(path, exclude_dirs=EXCLUDE_DIRS):
    root, subfolders, _ = next(os.walk(path))
    return { subfolder: os.path.join(root, subfolder) 
             for subfolder in subfolders 
             if subfolder not in exclude_dirs }


for ratio in ratios:
    embs_dir = glob(PATH + '/embeddings*ratio-{}'.format(ratio))
    graphs_dir = glob(PATH + '/graphs*ratio-{}'.format(ratio))
    
    # graphs_dir is a list, but it should only contain one folder
    rel_folders = collect_subfolders(graphs_dir[0])

    confs[ratio] = {
        'embs_dir': embs_dir,
        'rel_folders': rel_folders,
    }


def data_generator(ratio=20, dim=DIM, epochs=EPOCHS):
    """
    Iterate over folds (loops) and produce triples: emb file, train, test
    file

    """
    conf = confs[ratio]

    for fold in folds:
        emb_file = glob(conf['embs_dir'][0] + '/' + EMB_GLOB(fold, dim, epochs))

        for rel, rel_folder in conf['rel_folders'].items():
            tr_file = glob(rel_folder + '/' + TR_GLOB(fold))
            te_file = glob(rel_folder + '/' + TE_GLOB(fold))

            obj = {
                'fold': fold,
                'relation': rel,
                'ratio': ratio,
                'dim': dim,
                'epoch': epochs,
                'emb_path': emb_file[0], 
                'tr_path': tr_file[0], 
                'te_path': te_file[0],
            }

            yield obj


EUC_SVM_PARAM_GRID = { 
    'C': [0.1, 1, 10], 
    'max_iter': [100, 500],
    }


HYP_SVM_PARAM_GRID = { 
    'C': [0.1, 1, 10], 
    'batch_size': [16, 32], 
    'epochs': [500],
    'lr': [0.1, 0.001],
    'pretrained': [False, True],
    }
