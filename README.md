# Hyperbolic SVM

Python implementation of hyperbolic SVM, as introduced in `[1]`. This is a Python
adaptation of the official imlementation in Matlab `[2]`. This is the official code repository from this paper https://www.aclweb.org/anthology/W19-5805/ .

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

## Citation

If you use it and find useful please consider citing this paper https://www.aclweb.org/anthology/W19-5805/

```
@inproceedings{agibetov-etal-2019-using,
    title = "Using hyperbolic large-margin classifiers for biological link prediction",
    author = "Agibetov, Asan  and
      Dorffner, Georg  and
      Samwald, Matthias",
    booktitle = "Proceedings of the 5th Workshop on Semantic Deep Learning (SemDeep-5)",
    month = "12 " # aug,
    year = "2019",
    address = "Macau, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-5805",
    pages = "26--30",
}
```

## References

* [1] `https://arxiv.org/abs/1806.00437`
* [2] `https://github.com/hhcho/hyplinear`
* [3] `https://github.com/plumdeq/neuro-kglink`
