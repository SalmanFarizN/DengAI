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
                    "reanalysis_specific_humidity_g_per_kg",
                    "reanalysis_dew_point_temp_k",
                    "reanalysis_min_air_temp_k",
                    "station_min_temp_c",
                    "reanalysis_relative_humidity_percent",
                    "station_avg_temp_c",
                    "reanalysis_precip_amt_kg_per_m2",
                    "reanalysis_air_temp_k",
                    "reanalysis_sat_precip_amt_mm",
                    "reanalysis_avg_temp_k",
                    "station_max_temp_c",
                    "station_precip_mm",
                    "ndvi_sw",
                    "weekofyear_col",

                    "reanalysis_specific_humidity_g_per_kg_lag1",
                    "reanalysis_specific_humidity_g_per_kg_lag2",
                    "reanalysis_specific_humidity_g_per_kg_lag3",
                    "reanalysis_specific_humidity_g_per_kg_lag4",

                    "reanalysis_dew_point_temp_k_lag1",
                    "reanalysis_dew_point_temp_k_lag2",
                    "reanalysis_dew_point_temp_k_lag3",
                    "reanalysis_dew_point_temp_k_lag4",

                    "reanalysis_min_air_temp_k_lag1",
                    "reanalysis_min_air_temp_k_lag2",
                    "reanalysis_min_air_temp_k_lag3",
                    "reanalysis_min_air_temp_k_lag4",

                    "station_min_temp_c_lag1",
                    "station_min_temp_c_lag2",
                    "station_min_temp_c_lag3",
                    "station_min_temp_c_lag4",

                    "reanalysis_relative_humidity_percent_lag1",
                    "reanalysis_relative_humidity_percent_lag2",
                    "reanalysis_relative_humidity_percent_lag3",
                    "reanalysis_relative_humidity_percent_lag4",

                    "station_avg_temp_c_lag1",
                    "station_avg_temp_c_lag2",
                    "station_avg_temp_c_lag3",
                    "station_avg_temp_c_lag4",

                    "reanalysis_precip_amt_kg_per_m2_lag1",
                    "reanalysis_precip_amt_kg_per_m2_lag2",
                    "reanalysis_precip_amt_kg_per_m2_lag3",
                    "reanalysis_precip_amt_kg_per_m2_lag4",

                    "reanalysis_air_temp_k_lag1",
                    "reanalysis_air_temp_k_lag2",
                    "reanalysis_air_temp_k_lag3",
                    "reanalysis_air_temp_k_lag4",

                    "reanalysis_sat_precip_amt_mm_lag1",
                    "reanalysis_sat_precip_amt_mm_lag2",
                    "reanalysis_sat_precip_amt_mm_lag3",
                    "reanalysis_sat_precip_amt_mm_lag4",

                    "reanalysis_avg_temp_k_lag1",
                    "reanalysis_avg_temp_k_lag2",
                    "reanalysis_avg_temp_k_lag3",
                    "reanalysis_avg_temp_k_lag4",

                    "station_max_temp_c_lag1",
                    "station_max_temp_c_lag2",
                    "station_max_temp_c_lag3",
                    "station_max_temp_c_lag4",

                    "station_precip_mm_lag1",
                    "station_precip_mm_lag2",
                    "station_precip_mm_lag3",
                    "station_precip_mm_lag4",

                    "ndvi_sw_lag1",
                    "ndvi_sw_lag2",
                    "ndvi_sw_lag3",
                    "ndvi_sw_lag4",

                    "weekofyear_col_cosine",
                    "weekofyear_col_sine",

                    # "station_avg_temp_c_temp_composite",

                    # "reanalysis_air_temp_k_saturation_composite"
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