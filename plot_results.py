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


@click.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.argument('viz-params-file', type=click.Path(exists=True))
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

    if rel2name.viz_params['drop_na']:
        df = df.dropna()

    agg_df = df[['relation', 'train_roc_auc', 'test_roc_auc', 'algo']]\
                .groupby(['algo', 'relation']).agg('mean').reset_index()
    # flatten columns

    # cols = [' '.join(c).strip() for c in agg_df.columns]
    # agg_df.columns = cols
    
    euc_df = agg_df[agg_df['algo'] == 'euc_svm'].set_index('relation')
    hyp_df = agg_df[agg_df['algo'] == 'hyp_svm'].set_index('relation')

    euc_df = euc_df.sort_values('test_roc_auc')
    hyp_df = hyp_df.sort_values('test_roc_auc')

    logger.info('Euc SVM')
    logger.info('train roc auc {:.3f} +/- {:.3f}'.format(euc_df['train_roc_auc'].mean(), euc_df['train_roc_auc'].std()**2))
    logger.info('test roc auc {:.3f} +/- {:.3f}'.format(euc_df['test_roc_auc'].mean(), euc_df['test_roc_auc'].std()**2))
    logger.info('Hyp SVM')
    logger.info('train roc auc {:.3f} +/- {:.3f}'.format(hyp_df['train_roc_auc'].mean(), hyp_df['train_roc_auc'].std()**2))
    logger.info('test roc auc {:.3f} +/- {:.3f}'.format(hyp_df['test_roc_auc'].mean(), hyp_df['test_roc_auc'].std()**2))

    x_sorted = hyp_df.index

    fig, axes = plt.subplots(2, 1, sharey=True, sharex=True)
    # sns.barplot(x='relation', y='train_roc_auc', data=df, hue='algo', ax=axes[0], ci='sd')
    # sns.relplot(x=euc_df.loc[x_sorted]['train_roc_auc'], 
    #             y=hyp_df.loc[x_sorted]['train_roc_auc'], ax=axes[0])
    # axes[0].set_title('train performance')
    # sns.barplot(x='relation', y='test_roc_auc', data=df, hue='algo', ax=axes[1], ci='sd')
    # sns.relplot(x=euc_df.loc[x_sorted]['test_roc_auc'], 
    #             y=hyp_df.loc[x_sorted]['test_roc_auc'], ax=axes[1])
    # axes[1].set_title('test performance')

    sns.scatterplot(euc_df.loc[x_sorted]['train_roc_auc'], 
                    hyp_df.loc[x_sorted]['train_roc_auc'], 
                    ax=axes[0])
    axes[0].plot([0.0, 1.0], [0.0, 1.0], '--', linewidth=0.5)
    axes[0].set_title('train performance')
    axes[0].set_ylim([0.45, 1.1])
    axes[0].set_xlim([0.45, 1.1])

    sns.scatterplot(euc_df.loc[x_sorted]['test_roc_auc'], 
                    hyp_df.loc[x_sorted]['test_roc_auc'], 
                    ax=axes[1])
    axes[1].plot([0.0, 1.0], [0.0, 1.0], '--', linewidth=0.5)
    axes[1].set_title('test performance')
    axes[1].set_ylim([0.45, 1.1])
    axes[1].set_xlim([0.45, 1.1])

    # for ax in axes:
    #     xticks = ax.get_xticklabels()
    #     for tick in xticks:
    #         tick.set_rotation(60)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
