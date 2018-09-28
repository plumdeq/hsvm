#!/usr/bin/env python3
# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Plot results from a pickle file

"""
# Standard-library imports
import json
import pickle
import logging
import itertools as it
import functools as fun

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Third-party imports
import click
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

VIZ_PARAMS = '/media/Warehouse/nips2018-rel_learning/bio-kg/viz_params.json'


class Rel2Name():
    def __init__(self, viz_params_file):
        self.viz_params = None
        with open(viz_params_file, 'r') as f:
            self.viz_params = json.load(f)

        self.rel2idx = pd.read_csv(
                self.viz_params['rel2idx'], sep='\t', index_col=0, header=None)
        self.rel2name = pd.read_csv(
                self.viz_params['rel2name'], sep=' ', index_col=0, header=None)

    def get_name(self, relix):
        long_name = self.rel2idx.loc[int(relix)].item()
        short_name = self.rel2name.loc[long_name].item()

        return short_name


@click.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--viz-params-file', type=click.Path(exists=True), default=VIZ_PARAMS)
def main(results_file, viz_params_file):
    res_obj = None
    rel2name = Rel2Name(viz_params_file)

    with open(results_file, 'rb') as f:
        res_obj = pickle.load(f)

    all_records = it.chain(*[res_obj[key] for key in res_obj])
    df = pd.DataFrame(dict((key, value) for key, value in x.items()
                                        if key != 'train_params') 
                                        for x in all_records)


    df['relation'] = df['relation'].apply(rel2name.get_name)

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
    sns.barplot(x='relation', y='train_roc_auc', data=df, hue='algo', ax=axes[0], ci='sd')
    axes[0].set_title('train performance')
    sns.barplot(x='relation', y='test_roc_auc', data=df, hue='algo', ax=axes[1], ci='sd')
    axes[1].set_title('test performance')

    for ax in axes:
        xticks = ax.get_xticklabels()
        for tick in xticks:
            tick.set_rotation(60)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
