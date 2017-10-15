from __future__ import print_function

import math

import numpy as np
import pandas as pd

from data_transform_processor import DataTransformProcessor
from models.tree_models.lgbm import LGBM
from schema.core import PROPERTIES_RENAME_DICT, TRANSACTION_RENAME_DICT


class FeatureEngineer(object):
    """
    Class to fill missing values and create new features based on existing features.
    """

    properties_rename_dict = PROPERTIES_RENAME_DICT
    transaction_rename_dict = TRANSACTION_RENAME_DICT

    @classmethod
    def rename(cls, df, is_properties=True):
        """
        Rename raw properties table.
        :param df: raw feature df
        :param is_properties: rename properties df or transaction df
        :return: renamed feature df
        """
        if is_properties:
            return df.rename(columns=cls.properties_rename_dict)
        else:
            return df.rename(columns=cls.transaction_rename_dict)

    @classmethod
    def merge(cls, df_properties, df_transaction):
        """
        Merge.
        :param df_properties: 
        :param df_transaction: 
        :return: 
        """
        # check whether the df has been renamed, if not, rename it
        if 'id_parcel' not in df_properties:
            df_properties = cls.rename(df_properties, is_properties=True)
        if 'id_parcel' not in df_transaction:
            df_transaction = cls.rename(df_transaction, is_properties=False)

        return pd.merge(df_transaction, df_properties, on='id_parcel', how='left')

    @staticmethod
    def _fill_column(df, model, data_transform_processor):
        """
        Fill missing values for one column, use supervised model to learn from non-missing values
        :param df: 
        :param model: model used to train and predict
        :param data_transform_processor: data transform processor
        :return: feature column with filled values
        """
        idx_train = df[data_transform_processor.label_col].notnull().values

        X_all = data_transform_processor.pre_process(df)
        y_all = df[data_transform_processor.label_col].values

        X_train = data_transform_processor.fit_transform(X_all[idx_train])
        y_train = y_all[idx_train]

        model.fit(X_train, y_train)

        X_pred = data_transform_processor.transform(X_all[~idx_train])
        y_pred = model.predict(X_pred)

        y_all[~idx_train] = y_pred
        return y_all

    def fill_missing_value_model_based(self, df):
        """
        Use supervised model to learn features from non-missing values, 
        take too long to run, separate from original code
        :param df: 
        :return: 
        """
        # fill region variables based on spatial data
        region_col_lst = ['region_zip', 'region_neighbor', 'region_county', 'region_city']
        for y_col in region_col_lst:
            region_pred_lst = [col for col in region_col_lst if col != y_col]

            data_transform_processor = DataTransformProcessor(
                label_col=y_col,
                numerical_cols=['latitude', 'longitude'],
                categorical_cols=region_pred_lst,
            )

            idx_train = df[y_col].notnull().values
            params = {
                'max_bin': 80,
                'learning_rate': 0.0216,
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': len(np.unique(df.loc[idx_train, y_col])),
                'feature_fraction': 0.94,
                'bagging_fraction': 0.85,
                'bagging_freq': 80,
                'num_leaves': 110,
                'n_estimators': 200,
            }
            model = LGBM(
                is_classifier=True,
                categorical_feature=data_transform_processor.categorical_col_idx,
                **params
            )

            df[y_col] = self._fill_column(df, model, data_transform_processor)

        return df

    @staticmethod
    def fill_missing_value_stats(df, inplace=False):
        """
        High level function to fill missing values
        :param df: 
        :param inplace:
        :return: 
        """
        if not inplace:
            df = df.copy()

        # fill area pool
        area_pool_median = df.loc[df['num_pool'].notnull() & (df['area_pool'].notnull()), 'area_pool'].median()
        df.loc[df['num_pool'].notnull() & df['area_pool'].isnull(), 'area_pool_filled'] = area_pool_median
        df.loc[df['num_pool'].isnull(), 'area_pool_filled'] = 0

        return df

    @staticmethod
    def add_features_properties(df, inplace=False):
        """
        Add features based on existing features, only relates to properties.
        :param df: data frame for properties
        :param inplace: 
        :return: data frame with added features
        """
        if not inplace:
            df = df.copy()

        # assessed value ratios
        df['ratio_value_land_vs_building'] = df['value_land'] / df['value_building']
        df['ratio_value_tax_property_vs_total'] = df['value_tax_property'] / df['value_total']

        # area ratios
        df['ratio_area_living_vs_total'] = df['area_total_finished_calc'] / df['area_lot']

        # average values based on location and deviation away from average
        avg_lst = ['area_lot', 'area_total_finished_calc', 'value_building', 'value_tax_property', 'value_total']
        df_city_mean = df.groupby('region_city')[avg_lst].mean()
        df_zip_mean = df.groupby('region_zip')[avg_lst].mean()

        for col in avg_lst:
            # average values
            df['avg_city_{}'.format(col)] = df['region_city'].map(df_city_mean[col])
            df['avg_zip_{}'.format(col)] = df['region_zip'].map(df_zip_mean[col])

            # deviations
            df['deviation_city_{}'.format(col)] = (df[col] - df['avg_city_{}'.format(col)]) / df[
                'avg_city_{}'.format(col)]
            df['deviation_zip_{}'.format(col)] = (df[col] - df['avg_zip_{}'.format(col)]) / df['avg_zip_{}'.format(col)]

        # properties count in region
        for region in ['zip', 'city', 'neighbor']:
            df_region_count = df['region_{}'.format(region)].value_counts()
            df['count_properties_{}'.format(region)] = df['region_{}'.format(region)].map(df_region_count)

        # manipulation of lat and long
        df['latitude_cos'] = df['latitude'].apply(math.cos)
        df['longitude_cos'] = df['longitude'].apply(math.cos)
        df['latitude_times_longitude'] = df['latitude'] * df['longitude_cos']
        df['coordinate_x'] = 6371 * np.cos(df['latitude']) * np.cos(df['longitude'])
        df['coordinate_y'] = 6371 * np.cos(df['latitude']) * np.sin(df['longitude'])
        df['coordinate_z'] = 6371 * np.sin(df['latitude'])

        # polynomials of variables
        for col in ['area_lot', 'area_total_finished_calc', 'value_building', 'value_tax_property', 'value_total']:
            df['{}_square'.format(col)] = df[col] ** 2

        # variables related to rooms and areas
        df['area_room_avg'] = (df['area_total_finished_calc'] / df['num_room']).replace(np.inf, np.nan)

        df['num_extra_room'] = df['num_room'] - df['num_bedroom'] - df['num_bathroom']
        df['area_extra'] = df['area_lot'] - df['area_total_finished_calc']

        # flag for nan value in region variable, in case when we use filled values for these variables
        for region in ['zip', 'city', 'neighbor', 'county']:
            df['flag_nan_region_{}'.format(region)] = df['region_{}'.format(region)].isnull().astype(int)

        return df

    @staticmethod
    def add_features_transactions(df, inplace=False):
        """
        Add features based on existing features, only relates to transactions (like transaction month).
        :param df: data frame merged by transactions and properties
        :param inplace:
        :return: data frame with added features
        """
        if not inplace:
            df = df.copy()

        # transaction month
        df['transaction_month'] = df['date'].str.split('-', expand=True)[1].values

        # transaction year
        df['transaction_year'] = df['date'].str.split('-', expand=True)[0].values

        # whether the property has multiple sales record in given period
        # df['flag_multiple_sales'] = df['id_parcel'].duplicated().values.astype(int)

        return df


