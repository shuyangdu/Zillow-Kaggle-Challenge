from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

from models.model_flow import ModelFlow
from models.backtest import BackTest


class HyperParameterOpt(object):
    """
    class to do hyper-parameter optimization.
    """

    @staticmethod
    def transform_trials_vals(misc_vals):
        """
        Transform trials['misc']['vals'] dictionary to the same format as best, to be used in space_eval.
        >>> trials = HyperParameterOpt().trials
        >>> trials['misc']['vals'] = {'n_estimators': [20]}
        >>> HyperParameterOpt.transform_trials_vals(trials['misc']['vals'])
        >>> {'n_estimators': 20}
        :param misc_vals: original dictionary
        :return: transformed dictionary
        """
        d = misc_vals.copy()
        for key, val in d.items():
            d[key] = val[0]
        return d

    def __init__(self, model_class=None, data_processor=None, fixed_params=None,
                 search_space=None, max_evals=1, max_workers=1):
        """
        Constructor.
        :param model_class: model class to be optimized
        :param data_processor: data transform processor
        :param fixed_params: parameters needed to construct the model but not in the search space
        :param search_space: dictionary of parameters for the model and corresponding candidate choices
        :param max_evals: max evaluation times
        :param max_workers: number of processes for parallel
        """
        self.model_class = model_class
        self.data_processor = data_processor
        self.fixed_params = fixed_params
        self.search_space = search_space
        self.max_evals = max_evals
        self.max_workers = max_workers

        self.trials = Trials()
        self.best = None

    def _cross_validation(self, model, X, y, seed):
        """
        Do the k-fold cross validation for the model
        :param model: concrete model object
        :param X: raw feature numpy array after pre-processing
        :param y: numpy array of labels
        :param seed: seed for cross validation
        :return: averaged MAE
        """
        model_flow = ModelFlow(model=model, data_processor=self.data_processor)
        backtest = BackTest(model_flow=model_flow, max_workers=self.max_workers)
        mae = backtest.single_cv(X, y, seed=seed)

        results = {
            'loss': mae,
            'status': STATUS_OK,
        }

        return results

    def optimize(self, X, y, seed):
        """
        Optimize the hyper-parameter in search space
        :param X: raw feature numpy array after pre-processing
        :param y: numpy array of labels
        :param seed: values in [11, 12, 21, 22, 31, 32, 41, 42, 51, 52]
        :return: None
        """
        def objective(params):
            params.update(self.fixed_params)
            model = self.model_class(**params)
            return self._cross_validation(model, X, y, seed)

        self.best = fmin(
            objective,
            self.search_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=self.trials
        )

    @property
    def trial_results(self):
        """
        :return: data frame shows the trial results
        """
        df = pd.DataFrame()
        for trial in self.trials.trials:
            params_dict = space_eval(self.search_space, self.transform_trials_vals(trial['misc']['vals']))

            # check dict values
            for key, value in params_dict.items():
                if type(value) == list or type(value) == tuple:
                    params_dict[key] = str(value)

            df_params = pd.DataFrame(params_dict, index=[0])
            df_results = pd.DataFrame(trial['result'], index=[0])
            df = pd.concat([df, pd.concat([df_params, df_results], axis=1)], axis=0, ignore_index=True)

        df = df.sort_values('loss')
        df.index = range(len(df))
        # note, for parameters defined using hp.choice, the value here is the index in hp.choice
        return df


