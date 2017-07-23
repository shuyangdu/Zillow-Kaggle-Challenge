from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataProcessorBase(object):
    """
    Base data processor.
    """
    def __init__(self, is_train=False):
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
    def __init__(self, is_train=False):
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
    def __init__(self, is_train=False):
        super(DataProcessorCategorical, self).__init__(is_train=is_train)
        # dictionary to store label encoder for each column
        self.label_encoder_dict = {}

    def fill_nan(self, df):
        """
        Fill NaN with 'NaN' string since label encoding will treat each np.nan as different value
        :param df: 
        :return: filled NaN
        """
        # change data type to object
        df = df.astype(object)

        df[pd.isnull(df)] = 'NaN'
        return df

    def label_encode(self, df):
        """
        :param df: 
        :return: encoded data frame
        """
        for col in df.columns:
            le = LabelEncoder()
            le.fit(list(df[col]))
            self.label_encoder_dict[col] = le
            df.loc[:, col] = le.transform(list(df[col]))
        return df

    def dummy_encode(self, df):
        # ToDo: label encode categorical features
        pass

