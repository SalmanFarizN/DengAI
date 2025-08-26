from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selection transformer for dengue fever prediction models.

    This transformer selects a curated set of meteorological and environmental features
    that have been identified as relevant for dengue fever case prediction. The feature
    set includes current values, lagged features, and engineered temporal features.

    The default feature set includes:
    - Core meteorological features (temperature, humidity, precipitation)
    - Lagged versions of key features (1-4 week lags)
    - Engineered temporal features (cosine/sine transformations of week-of-year)
    - Vegetation index features (NDVI)

    Attributes
    ----------
    features : list of str
        List of feature column names to select from the input DataFrame.

    Examples
    --------
    >>> selector = FeatureSelector()  # Uses default features
    >>> selected_data = selector.transform(augmented_data)
    >>>
    >>> # Custom feature selection
    >>> custom_features = ['temp_feature', 'humidity_feature']
    >>> selector = FeatureSelector(features=custom_features)
    """

    def __init__(self, features=None):
        """Initialize the feature selector.

        Parameters
        ----------
        features : list of str, optional
            List of feature column names to select. If None, uses a default
            set of features optimized for dengue fever prediction including:
            - Primary meteorological features
            - 1-4 week lagged features for temporal patterns
            - Engineered temporal features (cosine/sine week transformations)
            - Vegetation indices
            Default is None.

        Notes
        -----
        The default feature set represents the result of feature engineering
        experiments and domain knowledge about dengue fever transmission patterns.
        Lagged features capture the temporal delay between environmental conditions
        and disease manifestation.
        """
        if features is None:

            features = [
                # Main Feature Set
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
                # Below we Lag all our main features to deal with delayed total cases and there effect on weather.
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
                # Here we use cosine and sine of weekofyear against year to better adjust for seasonal changes.
                "weekofyear_col_cosine",
                "weekofyear_col_sine",
                # "station_avg_temp_c_temp_composite",
                # "reanalysis_air_temp_k_saturation_composite"
            ]

        self.features = features

    def fit(self, X=None, y=None):
        """Fit the feature selector.

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
        self : FeatureSelector
            Returns self for method chaining compatibility.
        """
        return self

    def transform(self, dataframe):
        """Select specified features from the input DataFrame.

        This method extracts the predefined set of features from the input DataFrame,
        providing feature selection for downstream modeling. The method includes
        diagnostic printing to track feature availability and selection.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame containing augmented features from previous pipeline steps.
            Expected to contain all features specified in self.features.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the selected features, maintaining the same
            index as the input DataFrame. Shape: (n_samples, n_selected_features)

        Raises
        ------
        KeyError
            If any of the specified features are not found in the input DataFrame.

        Notes
        -----
        - Prints diagnostic information about available and selected features
        - Creates a copy of the selected data to avoid modifying the original
        - The selected features are typically the result of prior feature engineering

        Examples
        --------
        >>> selector = FeatureSelector()
        >>> selected_features = selector.transform(engineered_data)
        >>> print(f"Selected {selected_features.shape[1]} features from {dataframe.shape[1]}")
        """
        # Print the name of columns
        print(f"Columns in the dataframe: {dataframe.columns.tolist()}")
        # Select features we want

        dataframe_selected = dataframe[self.features].copy()
        # Print the name of columns after selection
        print(f"Columns after selection: {dataframe_selected.columns.tolist()}")

        return dataframe_selected
