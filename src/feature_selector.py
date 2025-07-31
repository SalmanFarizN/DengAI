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
                "ndvi_ne",
                "ndvi_nw",
                "ndvi_se",
                "ndvi_sw",
                "precipitation_amt_mm",
                "reanalysis_air_temp_k",
                "reanalysis_avg_temp_k",
                "reanalysis_dew_point_temp_k",
                "reanalysis_max_air_temp_k",
                "reanalysis_min_air_temp_k",
                "reanalysis_precip_amt_kg_per_m2",
                "reanalysis_relative_humidity_percent",
                "reanalysis_sat_precip_amt_mm",
                "reanalysis_specific_humidity_g_per_kg",
                "reanalysis_tdtr_k",
                "station_avg_temp_c",
                "station_diur_temp_rng_c",
                "station_max_temp_c",
                "station_min_temp_c",
                "station_precip_mm",
                "year_col",
                "weekofyear_col",
                "ndvi_ne_shift_3",
                "ndvi_nw_shift_3",
                "ndvi_se_shift_3",
                "ndvi_sw_shift_3",
                "precipitation_amt_mm_shift_3",
                "reanalysis_air_temp_k_shift_3",
                "reanalysis_avg_temp_k_shift_3",
                "reanalysis_dew_point_temp_k_shift_3",
                "reanalysis_max_air_temp_k_shift_3",
                "reanalysis_min_air_temp_k_shift_3",
                "reanalysis_precip_amt_kg_per_m2_shift_3",
                "reanalysis_relative_humidity_percent_shift_3",
                "reanalysis_sat_precip_amt_mm_shift_3",
                "reanalysis_specific_humidity_g_per_kg_shift_3",
                "reanalysis_tdtr_k_shift_3",
                "station_avg_temp_c_shift_3",
                "station_diur_temp_rng_c_shift_3",
                "station_max_temp_c_shift_3",
                "station_min_temp_c_shift_3",
                "station_precip_mm_shift_3",
                "ndvi_ne_shift_2",
                "ndvi_nw_shift_2",
                "ndvi_se_shift_2",
                "ndvi_sw_shift_2",
                "precipitation_amt_mm_shift_2",
                "reanalysis_air_temp_k_shift_2",
                "reanalysis_avg_temp_k_shift_2",
                "reanalysis_dew_point_temp_k_shift_2",
                "reanalysis_max_air_temp_k_shift_2",
                "reanalysis_min_air_temp_k_shift_2",
                "reanalysis_precip_amt_kg_per_m2_shift_2",
                "reanalysis_relative_humidity_percent_shift_2",
                "reanalysis_sat_precip_amt_mm_shift_2",
                "reanalysis_specific_humidity_g_per_kg_shift_2",
                "reanalysis_tdtr_k_shift_2",
                "station_avg_temp_c_shift_2",
                "station_diur_temp_rng_c_shift_2",
                "station_max_temp_c_shift_2",
                "station_min_temp_c_shift_2",
                "station_precip_mm_shift_2",
                "ndvi_ne_shift_1",
                "ndvi_nw_shift_1",
                "ndvi_se_shift_1",
                "ndvi_sw_shift_1",
                "precipitation_amt_mm_shift_1",
                "reanalysis_air_temp_k_shift_1",
                "reanalysis_avg_temp_k_shift_1",
                "reanalysis_dew_point_temp_k_shift_1",
                "reanalysis_max_air_temp_k_shift_1",
                "reanalysis_min_air_temp_k_shift_1",
                "reanalysis_precip_amt_kg_per_m2_shift_1",
                "reanalysis_relative_humidity_percent_shift_1",
                "reanalysis_sat_precip_amt_mm_shift_1",
                "reanalysis_specific_humidity_g_per_kg_shift_1",
                "reanalysis_tdtr_k_shift_1",
                "station_avg_temp_c_shift_1",
                "station_diur_temp_rng_c_shift_1",
                "station_max_temp_c_shift_1",
                "station_min_temp_c_shift_1",
                "station_precip_mm_shift_1",
                "year_col_shift_1",
                "weekofyear_col_cosine",
                "weekofyear_col_sine",
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