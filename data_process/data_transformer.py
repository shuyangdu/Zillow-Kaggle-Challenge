from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


DUMMY_ENCODE_MAX_NUM_VALS = 20


class TransformerBase(object):
    """
    Base data processor.
    """
    @classmethod
    def fill_nan(cls, df):
        pass

    def __init__(self):
        """
        :param
        """
        pass


class TransformerNumerical(TransformerBase):
    """
    Process numerical features.
    """

    @classmethod
    def fill_nan(cls, df):
        """
        Fill NaN with 0
        :param df: 
        :return: df
        """
        return df.fillna(0).astype(float)

    @classmethod
    def log(cls, df):
        """
        Take log.
        """
        return np.log(df + 1.0)

    def __init__(self, use_scale=False, use_pca=False):
        super(TransformerNumerical, self).__init__()
        self.use_scale = use_scale
        self.use_pca = use_pca

        self.scaler = StandardScaler()
        self.pca = PCA()

    def get_params(self, deep=False):
        """
        Get parameters for this estimator. For details, refer to sklearn base Estimator API.
        :param deep: 
        :return: dictionary of parameters
        """
        return {
            'use_scale': self.use_scale,
            'use_pca': self.use_pca,
        }

    def fit(self, X, y=None):
        """
        Fit numerical transformer.
        :param X: numerical feature numpy array
        :param y: placeholder for API consistency
        :return: None
        """
        if self.use_scale:
            self.scaler.fit(X)
        if self.use_pca:
            self.pca.fit(X)

    def transform(self, X):
        """
        Apply the fitted transformation.
        :param X: numerical feature numpy array
        :return: transformed numerical feature numpy array
        """
        if self.use_scale:
            X = self.scaler.transform(X)
        if self.use_pca:
            X = self.pca.transform(X)
        return X

    def fit_transform(self, X, y=None):
        """
        Fit then transform.
        :param X: numerical feature numpy array, after label encoding in pre-process
        :param y: placeholder for API consistency
        :return: transformed numerical feature numpy array
        """
        self.fit(X)
        return self.transform(X)


class TransformerCategorical(TransformerBase):
    """
    Process categorical features.
    """

    NaN = 'NaN'

    DUMMY_ENCODE_MAX_NUM_VALS = 20

    @classmethod
    def fill_nan(cls, df):
        """
        Fill NaN with 'NaN' string since label encoding will treat each np.nan as different value
        :param df: 
        :return: data frame
        """
        df = df.where(pd.notnull(df), other=cls.NaN)

        # for label encoding, must convert data type to string, otherwise after list() it will become float again,
        # and if we add 'NaN' during fitting but no 'NaN' in transform, it will have type error
        df = df.astype(str)
        return df

    # @classmethod
    # def label_encoding(cls, df):
    #     """
    #     Label encoding for whole data set, no need to handle train and test differently.
    #     :param df:
    #     :return: encoded data frame
    #     """
    #     for col in df.columns:
    #         le = LabelEncoder()
    #         df.loc[:, col] = le.fit_transform(list(df[col]))
    #         cls.label_encoder_dict[col] = le
    #     return df

    def __init__(self, use_dummy=False):
        super(TransformerCategorical, self).__init__()

        self.label_encoder_dict = {}
        self.categorical_value_dict = {}

        self.use_dummy = use_dummy

        self.dummy_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        # one thing to note, since OneHotEncoder here doesn't handle drop first column, we can only use
        # linear regression with penalty term instead of pure OLS

        self.dummy_encode_col_idx = None

    def get_params(self, deep=False):
        """
        Get parameters for this estimator. For details, refer to sklearn base Estimator API.
        :param deep: 
        :return: dictionary of parameters
        """
        return {
            'use_dummy': self.use_dummy,
        }

    def _collect_train_values(self, X):
        """
        Collect unique categorical value for each column, add NaN if there is none.
        :param X: numpy arrays features after pre-process
        :return: 
        """
        # only do this for label encoding
        for i in range(X.shape[1]):
            col = X[:, i]
            self.categorical_value_dict[i] = np.unique(col)
            if self.NaN not in self.categorical_value_dict[i]:
                self.categorical_value_dict[i] = np.append(self.categorical_value_dict[i], self.NaN)

    def _fill_test_values(self, X):
        """
        Fill unseen values in test set to be NaN.
        :param X: numpy arrays features after pre-process
        :return: 
        """
        for i in range(X.shape[1]):
            col = X[:, i]
            X[~np.in1d(col, self.categorical_value_dict[i]), i] = self.NaN
        return X

    def fit(self, X, y=None):
        """
        Fit categorical transformer.
        :param X: categorical feature numpy array, after label encoding in pre-process
        :param y: placeholder for API consistency
        :return: None
        """
        # fit label encoder
        self._collect_train_values(X)
        for i in range(X.shape[1]):
            le = LabelEncoder()
            le.fit(list(self.categorical_value_dict[i]))
            self.label_encoder_dict[i] = le

        # use dummy encoding if needed
        if self.use_dummy:
            # filter columns with too many unique values
            self.dummy_encode_col_idx = []
            for i in range(X.shape[1]):
                if len(np.unique(X[:, i])) <= self.DUMMY_ENCODE_MAX_NUM_VALS:
                    self.dummy_encode_col_idx.append(i)

            # fit one hot encoder
            self.dummy_encoder.fit(X[:, self.dummy_encode_col_idx])

    def transform(self, X):
        """
        Apply the fitted transformation.
        :param X: categorical feature numpy array
        :return: transformed categorical feature numpy array
        """
        # transform using label encoder
        X = self._fill_test_values(X)
        for i in range(X.shape[1]):
            X[:, i] = self.label_encoder_dict[i].transform(list(X[:, i]))

        if self.use_dummy:
            X = self.dummy_encoder.transform(X[:, self.dummy_encode_col_idx])
        return X

    def fit_transform(self, X, y=None):
        """
        Fit then transform.
        :param X: categorical feature numpy array, after label encoding in pre-process
        :param y: placeholder for API consistency
        :return: transformed categorical feature numpy array
        """
        self.fit(X)
        return self.transform(X)
