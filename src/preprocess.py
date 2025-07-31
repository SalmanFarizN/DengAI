from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


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
            "reanalysis_dew_point_temp_k",
            "station_avg_temp_c",
            "station_min_temp_c",
        ]
        dataframe_clean = dataframe[features].copy()

        # Print the name of columns in dataframe
        print(f"Columns after selection: {dataframe_clean.columns.tolist()}")

        # Fill missing values
        dataframe_clean = dataframe_clean.ffill()

        X = dataframe_clean

        return X
