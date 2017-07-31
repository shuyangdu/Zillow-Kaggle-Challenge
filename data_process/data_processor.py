from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold


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
    def __init__(self, is_train=True):
        super(DataProcessorCategorical, self).__init__(is_train=is_train)
        # dictionary to store label encoder for each column
        self.label_encoder_dict = {}
        self.numeric_encoder_dict = {}

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
        Label encoding.
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


