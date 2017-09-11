from __future__ import print_function
from copy import deepcopy


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
