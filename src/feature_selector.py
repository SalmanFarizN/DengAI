from sklearn.base import BaseEstimator, TransformerMixin
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select specific features from a pandas DataFrame.
    This class implements the scikit-learn BaseEstimator and TransformerMixin interfaces.
    It can be used in a scikit-learn pipeline for feature selection.
    """

    def __init__(self, features=None):
        """
        Args:
            features (list): List of column names to select. If None, will use default features.
        """
        if features is None:
            features = [
                "reanalysis_specific_humidity_g_per_kg_shift_3",
                "reanalysis_dew_point_temp_k_shift_3",
                "station_max_temp_c_shift_3",
                "station_min_temp_c_shift_3",
                "station_avg_temp_c_shift_3",
                "weekofyear_col_cosine",
                "weekofyear_col_sine",
                "reanalysis_air_temp_k_shift_3",
                "reanalysis_avg_temp_k_shift_3",
                "reanalysis_max_air_temp_k_shift_3",
                "reanalysis_min_air_temp_k_shift_3"
            ]
        self.features = features

    def fit(self, X=None, y=None):
        # No fitting necessary, so we return self
        return self

    def transform(self, dataframe):
        """Select specified features from the dataframe.
        Args:
            dataframe (pd.DataFrame): Input dataframe.
        Returns:
            pd.DataFrame: Dataframe with only selected features.
        """
        # Print the name of columns
        print(f"Columns in the dataframe: {dataframe.columns.tolist()}")
        # Select features we want

        dataframe_selected = dataframe[self.features].copy()
        # Print the name of columns after selection
        print(f"Columns after selection: {dataframe_selected.columns.tolist()}")

        return dataframe_selected