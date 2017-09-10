from __future__ import print_function

import pandas as pd
import os
import cPickle
from data_process.column_schema import (PROPERTIES_RENAME_DICT, TRANSACTION_RENAME_DICT,
                                        NUMERICAL_COLS, CATEGORICAL_COLS, LOG_COLS)
from data_process.data_processor import DataProcessorNumerical, DataProcessorCategorical


PATH = '/Users/shuyangdu/Desktop/ZillowChallenge/zillow-kaggle-challenge'

# ToDo: Deprecated, use DataProcessor in data_transform_processor.py instead


class DataProcessPipeline(object):
    """
    High level data process pipeline, will use data processor.
    """

    # class attribute, seeds for 10 cross validation
    cv_seeds = [11, 12, 21, 22, 31, 32, 41, 42, 51, 52]

    def __init__(self, label_col='logerror',
                 properties_rename_dict=PROPERTIES_RENAME_DICT,
                 transaction_rename_dict=TRANSACTION_RENAME_DICT,
                 numerical_cols=NUMERICAL_COLS,
                 categorical_cols=CATEGORICAL_COLS,
                 log_cols=LOG_COLS,
                 encode_mode='label', use_scale=False):
        """
        Constructor
        :param properties_rename_dict: dictionary to rename properties data frame
        :param transaction_rename_dict: dictionary to rename transaction data frame
        :param numerical_cols: numerical feature columns
        :param categorical_cols: categorical feature columns
        :param log_cols: columns to be taken log
        :param encode_mode: label, dummy or numeric encoding
        :param use_scale: scale numerical features or not
        """
        self.properties_rename_dict = properties_rename_dict
        self.transaction_rename_dict = transaction_rename_dict
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.log_cols = log_cols
        self.label_col = label_col

        self.encode_mode = encode_mode
        self.use_scale = use_scale

        self.processor_numerical = DataProcessorNumerical()
        self.processor_categorical = DataProcessorCategorical(encode_mode=encode_mode)

    @property
    def original_feature_cols(self):
        """
        :return: both numerical and categorical columns
        """
        return self.numerical_cols + self.categorical_cols

    @property
    def final_feature_cols(self):
        """
        :return: feature columns after encoding categorical features
        """
        return self.numerical_cols + self.processor_categorical.encoded_cols

    def _merge(self, df_properties, df_train):
        """
        Rename and merge properties features and train data frames.
        :param df_properties: 
        :param df_train: 
        :return: merged data frame
        """
        df_properties.rename(columns=self.properties_rename_dict, inplace=True)
        df_train.rename(columns=self.transaction_rename_dict, inplace=True)

        return pd.merge(df_train, df_properties, on='id_parcel', how='left')

    def _add_feature(self, df_merged):
        """
        Add features based on feature engineering.
        :param df_merged: data frame after calling _merge
        :return: data frame with added features
        """
        # add feature based on transaction date
        df_merged['transaction_month'] = df_merged['date'].str.split('-', expand=True)[1].values
        return df_merged

    def prepare_data(self, df_properties, df_train):
        """
        Merge and add features based on original properties and train data set
        :param df_properties: 
        :param df_train: 
        :return: data set for research
        """
        df_merged = self._merge(df_properties, df_train)
        df_merged = self._add_feature(df_merged)
        return df_merged

    def k_fold(self, df=None, seed=52):
        """
        Return the train and test split data frame or index based on the random seed
        :param df: 
        :param seed: values in [11, 12, 21, 22, 31, 32, 41, 42, 51, 52]
        :return: train and test data frames (iterator)
        """
        folds = cPickle.load(open(os.path.join(PATH, 'folds', '_folds_{}.p'.format(seed)), 'rb'))

        for train_index, test_index in folds:
            if df is None:
                yield train_index, test_index
            else:
                yield df.loc[train_index].copy(), df.loc[test_index].copy()

    def pre_process(self, df):
        """
        Fill NaN, take log
        :param df: 
        :return: 
        """
        df_numerical = df[self.numerical_cols]
        df_categorical = df[self.categorical_cols]

        df_numerical = self.processor_numerical.fill_nan(df_numerical)
        # take log for numerical values
        df_numerical.loc[:, self.log_cols] = self.processor_numerical.log(df_numerical[self.log_cols])

        df_categorical = self.processor_categorical.fill_nan(df_categorical)

        return pd.concat([df_numerical, df_categorical, df[self.label_col]], axis=1)

    def post_process(self, df, is_train):
        """
        Scale and other transform, different behavior for train and test.
        """
        df = df.copy()
        self.processor_numerical.is_train = is_train
        self.processor_categorical.is_train = is_train

        df_categorical = df[self.categorical_cols].copy()

        if self.encode_mode == 'label':
            df_categorical = self.processor_categorical.label_encode(df_categorical)
        elif self.encode_mode == 'numeric':
            df_categorical = self.processor_categorical.\
                numeric_encode(df[self.categorical_cols+[self.label_col]].copy(), self.label_col)
        else:
            df_categorical = self.processor_categorical.dummy_encode(df_categorical)

        # delete original categorical columns and concatenate encoded ones
        df = df.drop(self.categorical_cols, axis=1)
        df = pd.concat([df, df_categorical], axis=1)

        # scale numerical values
        if self.use_scale:
            df.loc[:, self.numerical_cols] = self.processor_numerical.scale(df[self.numerical_cols])

        return df
