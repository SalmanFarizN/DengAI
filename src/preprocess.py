from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class Preprocess(BaseEstimator, TransformerMixin):
    """Data preprocessing transformer for time series dengue fever data.
    
    This transformer handles missing value imputation specifically designed for
    time series epidemiological data. It uses backward filling (backfill) which
    is particularly suitable for time series data where future values can help
    estimate missing past values, especially in the context of lagged features.
    
    The class follows scikit-learn's transformer interface and can be integrated
    into machine learning pipelines for consistent preprocessing across training
    and prediction workflows.
    
    Methods
    -------
    fit(X, y)
        Fit the preprocessor (no-op for this stateless transformer).
    transform(dataframe)
        Clean the input DataFrame by handling missing values.
        
    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> preprocessor = Preprocess()
    >>> clean_data = preprocessor.transform(raw_data)
    >>> 
    >>> # In a pipeline
    >>> pipeline = Pipeline([('preprocess', Preprocess()), ('model', XGBRegressor())])
    """

    def __init__(self):
        """Initialize the preprocessor.
        
        This transformer is stateless and requires no parameters.
        """
        pass

    def fit(self, X=None, y=None):
        """Fit the preprocessor to the data.
        
        This is a stateless transformer, so no actual fitting is performed.
        This method is required by the scikit-learn transformer interface.
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Input features. Not used in this stateless transformer.
        y : array-like, optional
            Target values. Not used in this transformer.
            
        Returns
        -------
        self : Preprocess
            Returns self for method chaining compatibility.
        """
        return self

    def transform(self, dataframe):
        """Clean the DataFrame by imputing missing values using backward fill.
        
        This method applies backward fill (bfill) imputation which fills missing
        values using the next valid observation. This approach is particularly
        suitable for time series data with lagged features where future observations
        can provide meaningful estimates for missing past values.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame containing features with potential missing values.
            Expected to have a time series structure (sorted by time).
            
        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with missing values filled using backward fill.
            Maintains the same shape and index as the input DataFrame.
            
        Notes
        -----
        - Backward fill is used specifically for time series dengue data
        - This method handles both raw data NaNs and missing values from lagged features
        - The operation is performed in-place on a copy to avoid modifying original data
        
        Examples
        --------
        >>> preprocessor = Preprocess()
        >>> clean_data = preprocessor.transform(raw_data)
        >>> print(f"Missing values before: {raw_data.isnull().sum().sum()}")
        >>> print(f"Missing values after: {clean_data.isnull().sum().sum()}")
        """

        # Fill missing values
        X = (
            dataframe.bfill()
        )  # Backfill used here for Nans in raw data but also deals with lagged data's missing values.
        return X
