# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Scikit-learn compatible classifier for Hyperbolic SVM

"""
# Standard-library imports
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Third-party imports
from tqdm import tqdm
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

# Cross-library imports
import htools


class LinearHSVM(BaseEstimator, LinearClassifierMixin):
    """
    Hyperbolic SVM in the hyperboloid model

    """
    def __init__(self, C=1.0, epochs=1000, lr=0.001, batch_size=16,
                 pretrained=True, fit_intercept=False):
        args_dict = locals()

        for name, val in args_dict.items():
            if name =='self':
                continue
            setattr(self, name, val)


    def fit(self, X, y):
        """
        Fitting data is done via SGD in the hyperboloid model of hyperbolic
        space

        """
        X, y = check_X_y(X, y)

        if self.pretrained:
            logger.debug('pretraining with default linear svm')
            self.linsvm_ = LinearSVC(fit_intercept=self.fit_intercept).fit(X, y)
            self.coef_ = self.linsvm_.coef_
            self.intercept_ = self.linsvm_.intercept_
        else:
            self.coef_ = np.random.randn(1, X.shape[-1])
            self.intercept_ = 0.0

        # train HSVM
        self.coef_ = self.hyper_train(X, y)

        self.classes_ = unique_labels(y)

        return self


    def hyper_train(self, X, Y):
        """
        Train SVM in Hyperbolic space. We run manually stochastic gradient descent

        """
        N = len(Y)
        w = self.coef_
        lr = self.lr
        C = self.C
        epochs = self.epochs
        batch_size = self.batch_size
        not_feasible_counter = 0

        for e in range(epochs):
            perm = np.arange(N)
            random.shuffle(perm)
            sum_loss = 0

            for i in range(0, N, batch_size):
                x = X[perm[i:i+batch_size]]
                y = Y[perm[i:i+batch_size]]

                grad = htools.grad_fn(w, x, y, C)
                w = w - lr * grad

                if not htools.is_feasible(w):
                    not_feasible_counter += 1
                    logger.debug('not feasible ({} times)'.format(not_feasible_counter))
                    res = sp.optimize.minimize_scalar(
                        lambda alpha: np.sum((htools.project_weight(w, alpha) - w)**2))
                    alpha = res.x
                    w = htools.project_weight(w, alpha)

                    assert htools.is_feasible(w)

                obj = htools.obj_fn(w, x, y, C)
                
                sum_loss += obj.item()

            logger.debug('loss {}'.format(sum_loss))

        
        # y_true = Y.ravel()
        # preds = htools.mink_prod(X, w).ravel()
        # correct = sum((y_true * preds) > 0)
        # auc = roc_auc_score(y_true, preds)
        # logger.info('acc {} auc {}'.format(correct/N, auc))

        return w


    def predict(self, X):
        """
        Predict class labels

        """
        logger.info('calling predict')
        check_is_fitted(self, ['coef_'])
        X = check_array(X)

        preds = htools.mink_prod(X, self.coef_).ravel()
        preds[preds >= 0] = 1
        preds[preds < 0] = -1

        return preds


    def decision_function(self, X):
        """
        This function is used for ROC AUC

        """
        check_is_fitted(self, ['coef_'])
        X = check_array(X)

        preds = htools.mink_prod(X, self.coef_).ravel()

        return preds
