from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class Preprocess(BaseEstimator, TransformerMixin):
    """Preprocess data by cleaning and selecting relevant features. Inputs are a pandas DataFrame.
    This class implements the scikit-learn BaseEstimator and TransformerMixin interfaces.
    It can be used in a scikit-learn pipeline for preprocessing data before model training.
    """

    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        # No fitting necessary, so we return self
        return self

    def transform(self, dataframe):
        """Clean the dataframe by filling missing values and selecting relevant features.
        Args:
            dataframe (pd.DataFrame): Input dataframe to be cleaned.
        Returns:
            pd.DataFrame: Cleaned dataframe.
        """

        # Print the name of columns
        print(f"Columns in the dataframe: {dataframe.columns.tolist()}")

        # Select features we want
        features = [
            "reanalysis_specific_humidity_g_per_kg",
            "reanalysis_relative_humidity_percent",
            "reanalysis_dew_point_temp_k",
            "station_avg_temp_c",
            "station_min_temp_c",
            "weekofyear",
            "ndvi_se",
            "ndvi_sw",
            "ndvi_ne",
            "ndvi_nw",
        ]
        dataframe_clean = dataframe[features].copy()

        # Add Sin and Cos of the weekofyear column
        dataframe_clean["weekofyear_sin"] = (
            2 * 3.14159 * dataframe_clean["weekofyear"] / 52
        ).apply(lambda x: np.sin(x))
        dataframe_clean["weekofyear_cos"] = (
            2 * 3.14159 * dataframe_clean["weekofyear"] / 52
        ).apply(lambda x: np.cos(x))

        # Add lagged features t-1 for climate & weather related features
        weather_features = [
            "reanalysis_specific_humidity_g_per_kg",
            "reanalysis_relative_humidity_percent",
            "reanalysis_dew_point_temp_k",
            "station_avg_temp_c",
            "station_min_temp_c",
        ]
        for feature in weather_features:
            dataframe_clean[f"{feature}_t-1"] = dataframe_clean[feature].shift(1)

        # Print the name of columns in dataframe
        print(f"Columns after selection: {dataframe_clean.columns.tolist()}")

        # Fill missing values
        dataframe_clean = dataframe_clean.ffill()

        X = dataframe_clean

        return X
