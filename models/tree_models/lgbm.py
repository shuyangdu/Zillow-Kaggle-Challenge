from __future__ import print_function
from lightgbm import LGBMRegressor


class LGBM(object):
    """
    Class wrapper for LightGBM regressor.
    """
    def __init__(self, feature_name=None, categorical_feature=None, **model_params):
        """
        Constructor.
        :param feature_name: list of feature column names, could be original_feature_cols if using label encoding
        :param categorical_feature: list of categorical feature column names
        """
        self.model = LGBMRegressor(**model_params)
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature

    def fit(self, X, y):
        """
        Wrap LGBMRegressor fit.
        :param X: 
        :param y: 
        :return: None
        """
        self.model.fit(X=X, y=y, feature_name=self.feature_name, categorical_feature=self.categorical_feature)

    def predict(self, X):
        """
        Wrap LGBMRegressor predict.
        :param X: 
        :return: predictions
        """
        return self.model.predict(X=X)
