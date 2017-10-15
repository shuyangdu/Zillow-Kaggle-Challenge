from __future__ import print_function

import cPickle
import os

import numpy as np

from data_process.data_transformer import TransformerNumerical, TransformerCategorical
from schema.core import PROPERTIES_RENAME_DICT, TRANSACTION_RENAME_DICT
from schema.columns_original import NUMERICAL_COLS, CATEGORICAL_COLS, LOG_COLS, LABEL_COL

PATH = '/Users/shuyangdu/Desktop/ZillowChallenge/zillow-kaggle-challenge/folds_new'


class DataTransformProcessor(object):
    """
    Data process. Make use of TransformerNumerical and TransformerCategorical.
    """
    # class attribute, seeds for 10 cross validation
    # cv_seeds = [11, 12, 21, 22, 31, 32, 41, 42, 51, 52]
    cv_seeds = [12, 22, 32, 42, 52]

    # global class attribute
    properties_rename_dict = PROPERTIES_RENAME_DICT
    transaction_rename_dict = TRANSACTION_RENAME_DICT

    @classmethod
    def k_fold(cls, seed=52):
        """
        Return the train and test split index based on the random seed
        :param seed: values in [12, 22, 32, 42, 52]
        :return: train and test index (iterator)
        """
        folds = cPickle.load(open(os.path.join(PATH, '_folds_{}.p'.format(seed)), 'rb'))

        for train_index, test_index in folds:
            yield train_index, test_index

    def __init__(self, use_scale=False, use_pca=False, use_dummy=False,
                 numerical_cols=NUMERICAL_COLS, categorical_cols=CATEGORICAL_COLS,
                 log_cols=LOG_COLS, label_col=LABEL_COL):
        """
        Constructor
        :param use_scale: if true, scale numerical features
        :param use_pca: if true, pca transform numerical features
        :param use_dummy: if true, dummy encode categorical features
        :param numerical_cols: numerical feature columns
        :param categorical_cols: categorical feature columns
        :param log_cols: columns in numerical columns to be taken log
        :prams label_col: label column
        """
        self.transformer_numerical = TransformerNumerical(use_scale=use_scale, use_pca=use_pca)
        self.transformer_categorical = TransformerCategorical(use_dummy=use_dummy)

        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.log_cols = log_cols
        self.label_col = label_col

        self.numerical_col_idx = range(len(numerical_cols))
        self.categorical_col_idx = range(len(numerical_cols), len(numerical_cols) + len(categorical_cols))

        self.is_fitted = False

    def pre_process(self, df, fit=True):
        """
        Fill NaN, take log for certain numerical features, label encoding all categorical features
        :param df: raw data frame
        :param fit: if true, fit label encoder
        :return: feature numpy array
        """
        df_numerical = df[self.numerical_cols].copy()
        df_categorical = df[self.categorical_cols].copy()

        df_numerical = TransformerNumerical.fill_nan(df_numerical)
        df_numerical.loc[:, self.log_cols] = TransformerNumerical.log(df_numerical[self.log_cols])

        df_categorical = TransformerCategorical.fill_nan(df_categorical)

        X_numerical = df_numerical.values
        X_categorical = df_categorical.values

        # label encode
        if fit:
            self.transformer_categorical.fit_label_encode(X_categorical)
        X_categorical = self.transformer_categorical.transform_label_encode(X_categorical)

        return np.concatenate([X_numerical, X_categorical], axis=1)

    def get_params(self, deep=False):
        """
        Get parameters for this estimator. For details, refer to sklearn base Estimator API.
        :param deep: 
        :return: dictionary of parameters
        """
        params_dict = {
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'log_cols': self.log_cols,
            'label_col': self.label_col,
        }
        params_dict.update(self.transformer_numerical.get_params())
        params_dict.update(self.transformer_categorical.get_params())
        return params_dict

    def fit(self, X, y=None):
        """
        Fit data processor.
        :param X: feature numpy array, after pre-processing
        :param y: placeholder for API consistency
        :return: None
        """
        assert self.numerical_col_idx is not None, 'numerical_col_idx is None! Must use pre_process first!'
        assert self.categorical_col_idx is not None, 'categorical_col_idx is None! Must use pre_process first!'

        # fit numerical transformer
        self.transformer_numerical.fit(X[:, self.numerical_col_idx].astype(float))

        # fit categorical transformer
        self.transformer_categorical.fit(X[:, self.categorical_col_idx])

        self.is_fitted = True

    def transform(self, X):
        """
        Apply the fitted transformation.
        :param X: feature numpy array
        :return: transformed feature numpy array
        """
        assert self.is_fitted, 'Must fit the data processor before using transform!'

        X_numerical = self.transformer_numerical.transform(X[:, self.numerical_col_idx].astype(float))
        X_categorical = self.transformer_categorical.transform(X[:, self.categorical_col_idx])

        # note: self.categorical_col_idx does not reflect true categorical col after dummy encoding
        return np.concatenate([X_numerical, X_categorical], axis=1).astype(float)

    def fit_transform(self, X, y=None):
        """
        Fit then transform.
        :param X: feature numpy array, after pre-process
        :param y: placeholder for API consistency
        :return: transformed feature numpy array
        """
        self.fit(X)
        return self.transform(X)
