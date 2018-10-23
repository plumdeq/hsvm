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

## Requirements

see `requirements.txt`

* [1] `https://arxiv.org/abs/1806.00437`
* [2] `https://github.com/hhcho/hyplinear`
* [3] `https://github.com/plumdeq/neuro-kglink`
