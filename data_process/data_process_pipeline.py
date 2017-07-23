from __future__ import print_function

import pandas as pd
import os
import cPickle
from data_process.column_schema import PROPERTIES_RENAME_DICT, TRANSACTION_RENAME_DICT, NUMERICAL_COLS, CATEGORICAL_COLS
from data_process.data_processor import DataProcessorNumerical, DataProcessorCategorical


class DataProcessPipeline(object):
    """
    High level data process pipeline, will use data processor.
    """
    def __init__(self, label_col='logerror',
                 properties_rename_dict=PROPERTIES_RENAME_DICT,
                 transaction_rename_dict=TRANSACTION_RENAME_DICT,
                 numerical_cols=NUMERICAL_COLS,
                 categorical_cols=CATEGORICAL_COLS,
                 ):
        """
        Constructor
        :param properties_rename_dict: dictionary to rename properties data frame
        :param transaction_rename_dict: dictionary to rename transaction data frame
        :param numerical_cols: numerical feature columns
        :param categorical_cols: categorical feature columns
        """
        self.properties_rename_dict = properties_rename_dict
        self.transaction_rename_dict = transaction_rename_dict
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.label_col = label_col

        self.processor_numerical = DataProcessorNumerical()
        self.processor_categorical = DataProcessorCategorical()

    @property
    def feature_cols(self):
        """
        :return: both numerical and categorical columns
        """
        return self.numerical_cols + self.categorical_cols

    def merge(self, df_properties, df_train):
        """
        Rename and merge properties features and train data frames.
        :param df_properties: 
        :param df_train: 
        :return: merged data frame
        """
        df_properties.rename(columns=self.properties_rename_dict, inplace=True)
        df_train.rename(columns=self.transaction_rename_dict, inplace=True)

        return pd.merge(df_train, df_properties, on='id_parcel', how='left')

    def k_fold(self, df, seed):
        """
        Return the train and test split data frame based on the random seed
        :param df: 
        :param seed: values in [11, 12, 21, 22, 31, 32, 41, 42, 51, 52]
        :return: train and test data frames (iterator)
        """
        folds = cPickle.load(open(os.path.join(os.getcwd(), 'folds', '_folds_{}.p'.format(seed)), 'rb'))

        for train_index, test_index in folds:
            yield df.loc[train_index].copy(), df.loc[test_index].copy()

    def pre_process(self, df):
        """
        Fill NaN, label encode
        :param df: 
        :return: 
        """
        df_numerical = df[self.numerical_cols]
        df_categorical = df[self.categorical_cols]

        df_numerical = self.processor_numerical.fill_nan(df_numerical)
        # take log for numerical values
        df_numerical = self.processor_numerical.log(df_numerical)

        df_categorical = self.processor_categorical.fill_nan(df_categorical)
        df_categorical = self.processor_categorical.label_encode(df_categorical)

        return pd.concat([df_numerical, df_categorical, df[self.label_col]], axis=1)

    def post_process(self, df, is_train):
        """
        Scale and other transform, different behavior for train and test.
        """
        df = df.copy()
        self.processor_numerical.is_train = is_train
        self.processor_categorical.is_train = is_train

        # scale numerical values
        df.loc[:, self.numerical_cols] = self.processor_numerical.scale(df[self.numerical_cols])

        return df
