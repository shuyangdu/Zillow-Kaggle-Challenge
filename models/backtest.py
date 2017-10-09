from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from sklearn.metrics import mean_absolute_error
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import copy_reg
import types

from data_process.data_transform_processor import DataTransformProcessor


# work around for pickle object in ProcessPoolExecutor
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
copy_reg.pickle(types.MethodType, _pickle_method)


class BackTest(object):
    """
    Cross validation back test.
    """
    def __init__(self, model_flow=None, max_workers=2):
        """
        Constructor.
        :param model_flow: ModelFlow wrapper of model and data processor
        :param max_workers: max number of workers for parallel
        """
        self.model_flow = model_flow
        self.max_workers = max_workers
        self.num = 0

    def _fit_predict_eval(self, X_y_train_val):
        """
        Fit model flow on train set, predict on validation set and evaluate
        :param X_y_train_val: (X_train, y_train, X_val, y_val)
        :return: MAE in validation set
        """
        X_train, y_train, X_val, y_val = X_y_train_val

        self.model_flow.fit(X_train, y_train)
        y_pred = self.model_flow.predict(X_val)
        return mean_absolute_error(y_val, y_pred)

    def single_cv(self, X, y, seed=12):
        """
        Single run of cross validation.
        :param X: raw feature numpy array after pre-processing
        :param y: numpy array of labels
        :param seed: random seed for cross validation
        :return: mean MAE
        """
        # mae_lst = []
        # for train_idx, val_idx in DataTransformProcessor.k_fold(seed=seed):
        #     X_train, X_val = X[train_idx], X[val_idx]
        #     y_train, y_val = y[train_idx], y[val_idx]
        #
        #     self.model_flow.fit(X_train, y_train)
        #     y_pred = self.model_flow.predict(X_val)
        #     mae_lst.append(mean_absolute_error(y_val, y_pred))

        X_y_train_val_lst = []
        for train_idx, val_idx in DataTransformProcessor.k_fold(seed=seed):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            X_y_train_val_lst.append((X_train, y_train, X_val, y_val))

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            mae_lst = list(executor.map(self._fit_predict_eval, X_y_train_val_lst))

        print('Single CV finished')

        return np.mean(mae_lst)

    def full_cv(self, X, y):
        """
        Full run of 10 cross validations.
        :param X: raw feature numpy array after pre-processing
        :param y: numpy array of labels 
        :return: mean MAE
        """
        mae_lst = []
        for seed in DataTransformProcessor.cv_seeds:
            mae_lst.append(self.single_cv(X, y, seed=seed))
        return np.mean(mae_lst)
