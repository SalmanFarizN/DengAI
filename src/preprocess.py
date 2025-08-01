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


        X = dataframe.bfill()
        return X

