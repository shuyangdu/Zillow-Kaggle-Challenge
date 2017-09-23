from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from sklearn.metrics import mean_absolute_error

from data_process.data_transform_processor import DataTransformProcessor


class BackTest(object):
    """
    Cross validation back test.
    """
    def __init__(self, model_flow=None):
        """
        Constructor.
        :param model_flow: ModelFlow wrapper of model and data processor
        """
        self.model_flow = model_flow

    def single_cv(self, X, y, seed=52):
        """
        Single run of cross validation.
        :param X: raw feature numpy array after pre-processing
        :param y: numpy array of labels
        :param seed: random seed for cross validation
        :return: mean MAE
        """
        mae_lst = []
        for train_idx, val_idx in DataTransformProcessor.k_fold(seed=seed):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.model_flow.fit(X_train, y_train)
            y_pred = self.model_flow.predict(X_val)
            mae_lst.append(mean_absolute_error(y_val, y_pred))

        return np.mean(mae_lst)

    # def single_cv(self, df=None, seed=52):
    #     """
    #     Single run of cross validation.
    #     :param df: whole data set
    #     :param seed:
    #     :return: mean MAE
    #     """
    #     mae_lst = []
    #     for df_train, df_val in self.data_process_pipeline.k_fold(df, seed):
    #         df_train = self.data_process_pipeline.post_process(df_train, is_train=True)
    #         df_val = self.data_process_pipeline.post_process(df_val, is_train=False)
    #
    #         X_train = df_train[self.data_process_pipeline.final_feature_cols].values
    #         y_train = df_train[self.data_process_pipeline.label_col].values
    #         X_val = df_val[self.data_process_pipeline.final_feature_cols].values
    #         y_val = df_val[self.data_process_pipeline.label_col].values
    #
    #         self.model.fit(X_train, y_train)
    #         y_pred = self.model.predict(X_val)
    #         mae_lst.append(mean_absolute_error(y_val, y_pred))
    #
    #     return np.mean(mae_lst)

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
