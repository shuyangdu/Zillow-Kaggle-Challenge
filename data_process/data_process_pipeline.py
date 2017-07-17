from __future__ import print_function

import pandas as pd
import os
import cPickle
from data_process.column_schema import PROPERTIES_RENAME_DICT, TRANSACTION_RENAME_DICT, NUMERICAL_COLS, CATEGORICAL_COLS


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
        :param seed:
        :return: train and test data frames (iterator)
        """
        folds = cPickle.load(open(os.path.join(os.getcwd(), 'folds', '_folds_{}.p'.format(seed)), 'rb'))

        for train_index, test_index in folds:
            yield df.loc[train_index].copy(), df.loc[test_index].copy()
