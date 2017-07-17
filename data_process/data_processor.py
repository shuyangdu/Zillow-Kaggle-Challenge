from __future__ import print_function
import numpy as np
import pandas as pd


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

    def fill_nan(self, df):
        # ToDo: fill nan for numerical features based on each column's meaning
        pass


class DataProcessorCategorical(DataProcessorBase):
    """
    Process categorical features.
    """
    def __init__(self, is_train=False):
        super(DataProcessorCategorical, self).__init__(is_train=is_train)

    def fill_nan(self, df):
        # ToDo: fill nan for categorical features based on each column's meaning
        pass

    def label_encode(self, df):
        # ToDo: label encode categorical features
        pass

    def dummy_encode(self, df):
        # ToDo: label encode categorical features
        pass

