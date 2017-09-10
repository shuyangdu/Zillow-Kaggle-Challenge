from __future__ import print_function

import pandas as pd
import numpy as np
import os
import cPickle
from data_process.column_schema import (PROPERTIES_RENAME_DICT, TRANSACTION_RENAME_DICT,
                                        NUMERICAL_COLS, CATEGORICAL_COLS, LOG_COLS)
from data_process.data_transformer import TransformerNumerical, TransformerCategorical

PATH = '/Users/shuyangdu/Desktop/ZillowChallenge/zillow-kaggle-challenge'


class DataProcessor(object):
    """
    Data process. Make use of TransformerNumerical and TransformerCategorical.
    """
    # class attribute, seeds for 10 cross validation
    cv_seeds = [11, 12, 21, 22, 31, 32, 41, 42, 51, 52]

    # global class attribute
    properties_rename_dict = PROPERTIES_RENAME_DICT
    transaction_rename_dict = TRANSACTION_RENAME_DICT
    numerical_cols = NUMERICAL_COLS
    categorical_cols = CATEGORICAL_COLS
    log_cols = LOG_COLS

    @classmethod
    def _merge(cls, df_properties, df_train):
        """
        Rename and merge properties features and train data frames.
        :param df_properties: 
        :param df_train: 
        :return: merged data frame
        """
        df_properties.rename(columns=cls.properties_rename_dict, inplace=True)
        df_train.rename(columns=cls.transaction_rename_dict, inplace=True)

        return pd.merge(df_train, df_properties, on='id_parcel', how='left')

    @classmethod
    def _add_feature(cls, df_merged):
        """
        Add features based on feature engineering.
        :param df_merged: data frame after calling _merge
        :return: data frame with added features
        """
        # add feature based on transaction date
        df_merged['transaction_month'] = df_merged['date'].str.split('-', expand=True)[1].values
        return df_merged

    @classmethod
    def prepare_data(cls, df_properties, df_train):
        """
        Merge and add features based on original properties and train data set
        :param df_properties: 
        :param df_train: 
        :return: data set for research
        """
        df_merged = cls._merge(df_properties, df_train)
        df_merged = cls._add_feature(df_merged)
        return df_merged

    @classmethod
    def pre_process(cls, df):
        """
        Fill NaN, take log for certain numerical features, label encoding all categorical features
        :param df: raw data frame
        :return: feature numpy array
        """
        df_numerical = df[cls.numerical_cols].copy()
        df_categorical = df[cls.categorical_cols].copy()

        df_numerical = TransformerNumerical.fill_nan(df_numerical)
        df_numerical.loc[:, cls.log_cols] = TransformerNumerical.log(df_numerical[cls.log_cols])

        df_categorical = TransformerCategorical.fill_nan(df_categorical)
        df_categorical = TransformerCategorical.label_encoding(df_categorical)

        # calculate class attribute for future use, assume original feature cols are the same for all data processors
        cls.categorical_col_idx = range(df_categorical.shape[1])
        cls.numerical_col_idx = range(df_categorical.shape[1], df_categorical.shape[1]+df_numerical.shape[1])

        return np.concatenate([df_categorical.values, df_numerical.values], axis=1)

    @classmethod
    def k_fold(cls, seed=52):
        """
        Return the train and test split index based on the random seed
        :param seed: values in [11, 12, 21, 22, 31, 32, 41, 42, 51, 52]
        :return: train and test index (iterator)
        """
        folds = cPickle.load(open(os.path.join(PATH, 'folds', '_folds_{}.p'.format(seed)), 'rb'))

        for train_index, test_index in folds:
            yield train_index, test_index

    def __init__(self, use_scale=False, use_pca=False, use_dummy=False):
        """
        Constructor
        :param use_scale: if true, scale numerical features
        :param use_pca: if true, pca transform numerical features
        :param use_dummy: if true, dummy encode categorical features
        """
        self.transformer_numerical = TransformerNumerical(use_scale=use_scale, use_pca=use_pca)
        self.transformer_categorical = TransformerCategorical(use_dummy=use_dummy)

    def get_params(self, deep=False):
        """
        Get parameters for this estimator. For details, refer to sklearn base Estimator API.
        :param deep: 
        :return: dictionary of parameters
        """
        params_dict = {}
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
        # fit numerical transformer
        self.transformer_numerical.fit(X[:, self.numerical_col_idx])

        # fit categorical transformer
        self.transformer_categorical.fit(X[:, self.categorical_col_idx])

    def transform(self, X):
        """
        Apply the fitted transformation.
        :param X: feature numpy array
        :return: transformed feature numpy array
        """
        X_numerical = self.transformer_numerical.transform(X[:, self.numerical_col_idx])
        X_categorical = self.transformer_categorical.transform(X[:, self.categorical_col_idx])
        return np.concatenate([X_categorical, X_numerical], axis=1)

    def fit_transform(self, X, y=None):
        """
        Fit then transform.
        :param X: feature numpy array, after pre-process
        :param y: placeholder for API consistency
        :return: transformed feature numpy array
        """
        self.fit(X)
        return self.transform(X)
