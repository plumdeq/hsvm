# Hyperbolic SVM

Python implementation of hyperbolic SVM, as introduced in `[1]`. This is a Python
adaptation of the official imlementation in Matlab `[2]`. 

## Raison d'être

* Hyperbolic SVM compatible with `scikit-learn`, i.e., inherits from
  `BaseEstimator, LinearClassifierMixin` for an easier integration into
  `scikit-learn` pipelines
* Simple `matplotlib` visualizations of decision boundaries for both Euclidean
  and hyperbolic SVMs in 2 dimensions
* Integrates seemlessly with evaluation pipeline for knowledge graph embeddings
  as in `[3]`

## Usage

* `python3 train.py ./data/gaussian/data_002.mat --c 1`
    * will train Euclidean and hyperbolic SVM on data generated with hyperbolic
      Gaussian
* see `--help` of `evaluate_embeddings.py, plot_results.py, train.py` for
  further details

## Requirements

see `requirements.txt`

## References

* [1] `https://arxiv.org/abs/1806.00437`
* [2] `https://github.com/hhcho/hyplinear`
* [3] `https://github.com/plumdeq/neuro-kglink`
