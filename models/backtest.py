from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


class BackTest(object):
    """
    Cross validation back test.
    """
    # ToDo: change model to model flow, which has the same API as ensemble
    def __init__(self, model=None, data_process_pipeline=None):
        """
        Constructor.
        :param model:  
        :param data_process_pipeline: 
        """
        self.model = model
        self.data_process_pipeline = data_process_pipeline

    # ToDo: Rewrite the single cv for model flow (ensemble flow)
    # def single_cv(self, X, y, seed=52):
    #     """
    #
    #     :param X:
    #     :param y:
    #     :param seed:
    #     :return:
    #     """

    def single_cv(self, df=None, seed=52):
        """
        Single run of cross validation.
        :param df: whole data set
        :param seed: 
        :return: mean MAE
        """
        mae_lst = []
        for df_train, df_val in self.data_process_pipeline.k_fold(df, seed):
            df_train = self.data_process_pipeline.post_process(df_train, is_train=True)
            df_val = self.data_process_pipeline.post_process(df_val, is_train=False)

            X_train = df_train[self.data_process_pipeline.final_feature_cols].values
            y_train = df_train[self.data_process_pipeline.label_col].values
            X_val = df_val[self.data_process_pipeline.final_feature_cols].values
            y_val = df_val[self.data_process_pipeline.label_col].values

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            mae_lst.append(mean_absolute_error(y_val, y_pred))

        return np.mean(mae_lst)

    def full_cv(self, df=None):
        """
        Full run of 10 cross validations.
        :param df: 
        :return: mean MAE
        """
        mae_lst = []
        for seed in self.data_process_pipeline.cv_seeds:
            mae_lst.append(self.single_cv(df, seed))
        return np.mean(mae_lst)
