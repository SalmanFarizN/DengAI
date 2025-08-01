from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
class Preprocess(BaseEstimator, TransformerMixin):
    """Preprocess data by cleaning. Inputs are a pandas DataFrame.
    This class implements the scikit-learn BaseEstimator and TransformerMixin interfaces.
    It can be used in a scikit-learn pipeline for preprocessing data before model training.
    """

    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        # No fitting necessary, so we return self
        return self

    def transform(self, dataframe):
        """Clean the dataframe by filling missing values.
        Args:
            dataframe (pd.DataFrame): Input dataframe to be cleaned.
        Returns:
            pd.DataFrame: Cleaned dataframe.
        """

        # Fill missing values
        dataframe_clean = self.fill_shift_nans_bfill(dataframe)

        X = dataframe_clean

        return X

    def fill_shift_nans_bfill(self,df):
        """
        Use backward fill - fills NaNs with next valid value
        Perfect for your 3-week shift situation!
        """
        df_filled = df.copy()

        # Sort properly first
        df_filled = df_filled.sort_values(['city', 'year', 'weekofyear'])

        # Backward fill within each city group
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'total_cases']

        df_filled[numeric_cols] = (
            df_filled.groupby('city')[numeric_cols]
            .bfill()  # Backward fill - uses next valid value
        )

        return df_filled

