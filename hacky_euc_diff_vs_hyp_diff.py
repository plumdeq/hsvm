#!/usr/bin/env python3
# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Plot results from pickle files to show differences in euclidean and mobius
differences for the same graph file

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

class Rel2Name():
    def __init__(self, viz_params_file):
        self.viz_params = None
        with open(viz_params_file, 'r') as f:
            self.viz_params = json.load(f)

        self.rel2idx = pd.read_csv(
                self.viz_params['rel2idx'], sep='\t', index_col=0, header=None)

        if self.viz_params['use_rel2name']:
            self.rel2name = pd.read_csv(
                    self.viz_params['rel2name'], sep=' ', index_col=0, header=None)

    def get_name(self, relix):
        long_name = self.rel2idx.loc[int(relix)].item()

        if self.viz_params['use_rel2name']:
            short_name = self.rel2name.loc[long_name].item()
        else:
            short_name = long_name

        return short_name


def to_df(results_file, rel2name):
    """Prepare a dataframe from the pickled results object"""
    with open(results_file, 'rb') as f:
        res_obj = pickle.load(f)

    all_records = it.chain(*[res_obj[key] for key in res_obj])
    df = pd.DataFrame(dict((key, value) for key, value in x.items()
                                        if key != 'train_params') 
                                        for x in all_records)

    df['relation'] = df['relation'].apply(rel2name.get_name)

    if rel2name.viz_params['drop_na']:
        df = df.dropna()

    agg_df = df[['relation', 'train_roc_auc', 'test_roc_auc', 'algo']]\
                .groupby(['algo', 'relation']).agg('mean').reset_index()
    # flatten columns

    # cols = [' '.join(c).strip() for c in agg_df.columns]
    # agg_df.columns = cols

    return agg_df


@click.command()
@click.argument('euc_results_file', type=click.Path(exists=True))
@click.argument('mobius_results_file', type=click.Path(exists=True))
@click.argument('viz-params-file', type=click.Path(exists=True))
def main(euc_results_file, mobius_results_file, viz_params_file):
    euc_res_obj, mobius_res_obj = None, None
    rel2name = Rel2Name(viz_params_file)

    euc_df = to_df(euc_results_file, rel2name)
    hyp_df = to_df(mobius_results_file, rel2name)

    euc_diff_df = euc_df[euc_df['algo'] == 'hyp_svm'].set_index('relation')
    mobius_diff_df = hyp_df[hyp_df['algo'] == 'hyp_svm'].set_index('relation')

    mobius_diff_df = mobius_diff_df.sort_values('test_roc_auc')
    x_sorted = mobius_diff_df.index

    fig, axes = plt.subplots(2, 1, sharey=True, sharex=True)
    sns.scatterplot(euc_diff_df.loc[x_sorted]['train_roc_auc'], 
                    mobius_diff_df.loc[x_sorted]['train_roc_auc'], 
                    ax=axes[0])
    axes[0].plot([0.0, 1.0], [0.0, 1.0], '--', linewidth=0.5)
    axes[0].set_title('train performance')
    axes[0].set_ylim([0.45, 1.1])
    axes[0].set_xlim([0.45, 1.1])

    sns.scatterplot(euc_diff_df.loc[x_sorted]['test_roc_auc'], 
                    mobius_diff_df.loc[x_sorted]['test_roc_auc'], 
                    ax=axes[1])
    axes[1].plot([0.0, 1.0], [0.0, 1.0], '--', linewidth=0.5)
    axes[1].set_title('test performance')
    axes[1].set_ylim([0.45, 1.1])
    axes[1].set_xlim([0.45, 1.1])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
