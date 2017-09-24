from __future__ import print_function
from copy import deepcopy

import pandas as pd


class ModelFlow(object):
    """
    Wrapper for Model and Data Processor, the same API as ensemble learner in ML-ensemble.
    """
    def __init__(self, model=None, data_processor=None):
        """
        Constructor.
        :param model: model object 
        :param data_processor: data processor object
        """
        self.model = model

        # only deepcopy data processor since model may be TensorFlow object, which cannot be deep copied
        self.data_processor = deepcopy(data_processor)

    def fit(self, X, y):
        """
        Fit both data processor and model
        :param X: raw feature numpy array after pre-processing
        :param y: numpy array of labels
        :return: None
        """
        # fit data processor
        X = self.data_processor.fit_transform(X)

        # fit model
        self.model.fit(X, y)

    def predict(self, X):
        """
        Transform X using data processor and predict using model
        :param X: raw feature numpy array after pre-processing
        :return: predictions
        """
        # transform X
        X = self.data_processor.transform(X)

        # predict
        return self.model.predict(X)

    @property
    def feature_importance_df(self):
        """
        Data Frame of feature importance for each feature, can only be used when model is LightGBM family models
        :return:
        """
        df = pd.DataFrame(
            self.model.model.feature_importances_,
            index=self.data_processor.categorical_cols + self.data_processor.numerical_cols,
            columns=['feature_importance'],
        )

        df.sort_values('feature_importance', ascending=False, inplace=True)

        return df

    @property
    def feature_importance_plot(self):
        """
        Plot the feature importance graph.
        :return: 
        """
        return self.feature_importance_df.plot(kind='barh', figsize=(10, 16))
