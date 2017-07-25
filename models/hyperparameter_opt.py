from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from hyperopt import fmin, tpe, Trials, STATUS_OK


class HyperParameterOpt(object):
    """
    Class to do hyper-parameter optimization.
    """
    def __init__(self, model_class=None, data_process_pipeline=None, df=None, fixed_params=None,
                 search_space=None, max_evals=1):
        """
        Constructor.
        :param model_class: model class to be optimized
        :param data_process_pipeline: data process pipeline
        :param df: data frame after pre-processing
        :param fixed_params: parameters needed to construct the model but not in the search space
        :param search_space: dictionary of parameters for the model and corresponding candidate choices
        :param max_evals: max evaluation times
        """
        self.model_class = model_class
        self.data_process_pipeline = data_process_pipeline
        self.df = df
        self.fixed_params = fixed_params
        self.search_space = search_space
        self.max_evals = max_evals

        self.trials = Trials()
        self.best = None

    def _cross_validation(self, model, seed):
        """
        Do the k-fold cross validation for the model
        :param model: concrete model object
        :return: averaged MAE
        """
        mae_lst = []
        for df_train, df_val in self.data_process_pipeline.k_fold(self.df, seed):
            df_train = self.data_process_pipeline.post_process(df_train, is_train=True)
            df_val = self.data_process_pipeline.post_process(df_val, is_train=False)

            X_train = df_train[self.data_process_pipeline.numerical_cols].values
            y_train = df_train[self.data_process_pipeline.label_col].values
            X_val = df_val[self.data_process_pipeline.numerical_cols].values
            y_val = df_val[self.data_process_pipeline.label_col].values

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mae_lst.append(mean_absolute_error(y_val, y_pred))

        results = {'loss': np.mean(mae_lst),
                   'status': STATUS_OK,
                   'loss_std': np.std(mae_lst)}

        return results

    def optimize(self, seed):
        """
        Optimize the hyper-parameter in search space
        :param seed: values in [11, 12, 21, 22, 31, 32, 41, 42, 51, 52]
        :return: None
        """
        def objective(params):
            params.update(self.fixed_params)
            model = self.model_class(**params)
            return self._cross_validation(model, seed)

        self.best = fmin(objective, self.search_space, algo=tpe.suggest,
                         max_evals=self.max_evals, trials=self.trials)

    @property
    def trial_results(self):
        """
        :return: data frame shows the trial results
        """
        df = pd.DataFrame()
        for trial in self.trials.trials:
            df_params = pd.DataFrame(trial['misc']['vals'])
            df_results = pd.DataFrame(trial['result'], index=[0])
            df = pd.concat([df, pd.concat([df_params, df_results], axis=1)], axis=0)

        # note, for parameters defined using hp.choice, the value here is the index in hp.choice
        return df.sort_values('loss')
