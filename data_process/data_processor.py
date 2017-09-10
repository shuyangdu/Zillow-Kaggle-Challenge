from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold


DUMMY_ENCODE_MAX_NUM_VALS = 20

# ToDo: Deprecated, use DataTransformer in data_transformer.py instead


class DataProcessorBase(object):
    """
    Base data processor.
    """
    def __init__(self, is_train=True):
        """
        :param is_train: train or test
        """
        self.is_train = is_train

    def fill_nan(self, df):
        pass


class DataProcessorNumerical(DataProcessorBase):
    """
    Process numerical features.
    """
    def __init__(self, is_train=True):
        super(DataProcessorNumerical, self).__init__(is_train=is_train)
        self.scaler = StandardScaler()

    def fill_nan(self, df):
        """
        Fill NaN with 0
        :param df: 
        :return: df
        """
        return df.fillna(0)

    def log(self, df):
        """
        Take log.
        """
        return np.log(df + 1.0)

    def scale(self, df):
        """
        Scale features to be mean 0 and std 1.
        """
        if self.is_train:
            self.scaler.fit(df)
        return self.scaler.transform(df)

    def pca_transform(self, df):
        # ToDo: pca
        pass


class DataProcessorCategorical(DataProcessorBase):
    """
    Process categorical features.
    """
    def __init__(self, is_train=True, encode_mode=None):
        super(DataProcessorCategorical, self).__init__(is_train=is_train)
        # label, dummy, numeric
        self.encode_mode = encode_mode

        # dictionary to store label encoder for each column
        self.label_encoder_dict = {}
        self.numeric_encoder_dict = {}

        # mark for NaN
        self.NaN = 'NaN'

        # dictionary to hold categorical columns and unique values in that column
        self.categorical_value_dict = {}

        # encoded columns for dummy encoding
        self.encoded_cols = []

    def fill_nan(self, df):
        """
        Fill NaN with 'NaN' string since label encoding will treat each np.nan as different value
        :param df: 
        :return: filled NaN
        """
        # change data type to object
        if self.encode_mode == 'label':
            df = df.where(pd.notnull(df), other=self.NaN)
            # for label encoding, must convert data type to string, otherwise after list() it will become float again,
            # and if we add 'NaN' during fitting but no 'NaN' in transform, it will have type error
            df = df.astype(str)
        else:
            df = df.astype(object)
        return df

    def _collect_train_values(self, df):
        """
        Collect unique categorical value for each column, add NaN if there is none.
        :param df: 
        :return: 
        """
        # only do this for label encoding
        for col in df.columns:
            self.categorical_value_dict[col] = df[col].unique()
            if self.encode_mode == 'label' and np.NaN not in self.categorical_value_dict[col]:
                self.categorical_value_dict[col] = np.append(self.categorical_value_dict[col], self.NaN)
            elif self.encode_mode == 'dummy' and pd.notnull(self.categorical_value_dict[col]).all():
                self.categorical_value_dict[col] = np.append(self.categorical_value_dict[col], np.nan)

    def _fill_test_values(self, df):
        """
        Fill unseen values in test set to be NaN.
        :param df: 
        :return: 
        """
        for col in df.columns:
            if self.encode_mode == 'dummy':
                df.loc[~df[col].isin(self.categorical_value_dict[col]), col] = np.nan
            elif self.encode_mode == 'label':
                df.loc[~df[col].isin(self.categorical_value_dict[col]), col] = self.NaN
        return df

    def label_encode(self, df):
        """
        Label encoding.
        """
        # fit encoder during training
        if self.is_train:
            self._collect_train_values(df)
            for col in df.columns:
                le = LabelEncoder()
                le.fit(list(self.categorical_value_dict[col]))
                self.label_encoder_dict[col] = le
            # store encoded columns
            self.encoded_cols = list(df.columns)
        # test
        else:
            df = self._fill_test_values(df)

        for col in df.columns:
            df.loc[:, col] = self.label_encoder_dict[col].transform(list(df[col]))
        return df

    def dummy_encode(self, df):
        """
        One hot encode.
        :param df: 
        :return: 
        """
        # filter columns with too many unique values
        columns = list(df.columns)
        for col in columns:
            if len(df[col].unique()) > DUMMY_ENCODE_MAX_NUM_VALS:
                df.drop(col, inplace=True, axis=1)

        # during training
        if self.is_train:
            self._collect_train_values(df)
            # no need to drop first since NaN is one of the type, which will automatically be dropped
            df = pd.get_dummies(df)

            self.encoded_cols = list(df.columns)
            return df

        # during test
        else:
            # deal with labels unseen in train set
            df = self._fill_test_values(df)
            # no need to drop first since NaN is one of the type, which will automatically be dropped
            df = pd.get_dummies(df)

            # deal with values in train set but not in test
            for col in self.encoded_cols:
                if col not in df.columns:
                    df.loc[:, col] = 0

            # reorder columns to be consistent with train set
            return df[self.encoded_cols]

    def numeric_encode(self, df, y_col):
        """
        Encode categorical features by statistics of dependent variables of corresponding category
        :param df: data frame including categorical and y columns
        :param y_col: dependent variable column name
        :return: encoded feature data frame, exclude y column
        """
        x_cols = [col for col in df.columns if col != y_col]
        if self.is_train:
            for col in x_cols:
                group = df.groupby(col, as_index=False)
                group_category = group.count()[col]
                group_mean = group.mean()[y_col]
                group_sum = group.sum()[y_col]
                group_count = group.count()[y_col]
                # leave one out mean
                group_benchmark = (group_sum.sum() - group_sum) / (group_count.sum() - group_count)
                self.numeric_encoder_dict[col] = pd.concat([group_category, group_mean - group_benchmark], axis=1)

        for col in self.numeric_encoder_dict.keys():
            df_numeric = self.numeric_encoder_dict[col]
            df[col] = pd.merge(df[[col]], df_numeric, how='left', on=col)[y_col].values

        # fill na if some category not showed in the training set
        df = df.fillna(df.mean().to_dict())

        return df[x_cols]
