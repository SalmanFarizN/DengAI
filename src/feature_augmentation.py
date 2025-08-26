from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class FeatureAugmentation(BaseEstimator, TransformerMixin):
    """Feature engineering transformer for time series epidemiological data.

    This transformer creates derived features through various augmentation techniques
    including temporal lagging, trigonometric transformations for seasonality,
    and composite feature engineering. It is specifically designed for dengue fever
    prediction where temporal patterns and seasonal effects are critical.

    The transformer supports multiple augmentation types:
    - Temporal lagging: Create lagged versions of features (1-4 weeks)
    - Trigonometric features: Cosine/sine transformations for seasonal patterns
    - Composite features: Domain-specific feature combinations

    Attributes
    ----------
    augmentations : list of dict
        Configuration for feature augmentations. Each dictionary specifies
        the augmentation type, target column, and parameters.

    Examples
    --------
    >>> augmenter = FeatureAugmentation()  # Uses default configuration
    >>> augmented_data = augmenter.transform(raw_data)
    >>>
    >>> # Custom augmentation configuration
    >>> custom_config = [{'type': 'shift', 'column': 'temperature', 'shift_periods': [1, 2]}]
    >>> augmenter = FeatureAugmentation(augmentations=custom_config)
    """

    def __init__(self, augmentations=None):
        """Initialize the feature augmentation transformer.

        Parameters
        ----------
        augmentations : list of dict, optional
            Configuration for feature augmentations. Each dictionary should contain:

            - 'type' : str
                Type of augmentation. Supported values:
                - 'shift': Create lagged versions of features
                - 'shift_log': Create log-transformed lagged features (experimental)
                - 'cosine': Create cosine transformation for seasonality
                - 'sine': Create sine transformation for seasonality
                - 'temp_composite': Temperature-based composite features
                - 'saturation_composite': Saturation deficit composite features

            - 'column' : str
                Name of the source column to augment

            - 'shift_periods' : int or list of int
                Number of periods to shift (for shift-based augmentations)

            If None, uses a comprehensive default configuration optimized for
            dengue fever prediction. Default is None.

        Notes
        -----
        The default configuration includes lagged features for all major
        meteorological variables, seasonal transformations, and has placeholders
        for experimental composite features that showed limited impact.
        """
        if augmentations is None:
            # Default augmentation for backward compatibility
            augmentations = [
                {
                    "type": "shift",
                    "column": "reanalysis_specific_humidity_g_per_kg",
                    "shift_periods": [1, 2, 3, 4],
                },
                {
                    "type": "shift",
                    "column": "reanalysis_dew_point_temp_k",
                    "shift_periods": [1, 2, 3, 4],
                },
                {
                    "type": "shift",
                    "column": "reanalysis_min_air_temp_k",
                    "shift_periods": [1, 2, 3, 4],
                },
                {
                    "type": "shift",
                    "column": "station_min_temp_c",
                    "shift_periods": [1, 2, 3, 4],
                },
                {
                    "type": "shift",
                    "column": "reanalysis_relative_humidity_percent",
                    "shift_periods": [1, 2, 3, 4],
                },
                {
                    "type": "shift",
                    "column": "station_avg_temp_c",
                    "shift_periods": [1, 2, 3, 4],
                },
                {
                    "type": "shift",
                    "column": "reanalysis_precip_amt_kg_per_m2",
                    "shift_periods": [1, 2, 3, 4],
                },
                {
                    "type": "shift",
                    "column": "reanalysis_air_temp_k",
                    "shift_periods": [1, 2, 3, 4],
                },
                {
                    "type": "shift",
                    "column": "reanalysis_sat_precip_amt_mm",
                    "shift_periods": [1, 2, 3, 4],
                },
                {
                    "type": "shift",
                    "column": "reanalysis_avg_temp_k",
                    "shift_periods": [1, 2, 3, 4],
                },
                {
                    "type": "shift",
                    "column": "station_max_temp_c",
                    "shift_periods": [1, 2, 3, 4],
                },
                {
                    "type": "shift",
                    "column": "station_precip_mm",
                    "shift_periods": [1, 2, 3, 4],
                },
                {"type": "shift", "column": "ndvi_sw", "shift_periods": [1, 2, 3, 4]},
                {"type": "cosine", "column": "weekofyear_col", "shift_periods": 0},
                {"type": "sine", "column": "weekofyear_col", "shift_periods": 0},
                # NOTE: We tried these composite features but they had little impact
                # {'type': 'temp_composite', 'column': 'station_avg_temp_c', 'shift_periods': 3},
                # {'type': 'saturation_composite', 'column': 'reanalysis_air_temp_k', 'shift_periods': 2}
            ]

        self.augmentations = augmentations

    def fit(self, X=None, y=None):
        """Fit the feature augmentation transformer.

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
        self : FeatureAugmentation
            Returns self for method chaining compatibility.
        """
        return self

    def transform(self, dataframe):
        """Apply feature augmentation to create derived temporal and seasonal features.

        This method processes the input DataFrame according to the configured
        augmentation specifications, creating lagged features, seasonal transformations,
        and composite features that capture temporal patterns relevant to disease spread.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame with base features. Expected to have temporal structure
            with features like meteorological variables and temporal indices.

        Returns
        -------
        pd.DataFrame
            Enhanced DataFrame with original features plus augmented features.
            New features follow naming conventions like:
            - '{feature}_lag{n}' for lagged features
            - '{feature}_cosine' for cosine seasonal transformations
            - '{feature}_sine' for sine seasonal transformations

        Notes
        -----
        - Each augmentation is applied sequentially
        - Missing columns in augmentation config will generate warnings
        - The method includes progress reporting for feature creation
        - Lagged features may introduce NaN values at time series boundaries

        Examples
        --------
        >>> augmenter = FeatureAugmentation()
        >>> augmented = augmenter.transform(base_features)
        >>> print(f"Features expanded from {base_features.shape[1]} to {augmented.shape[1]}")
        """
        # Make a copy to avoid modifying the original
        dataframe_augmented = dataframe.copy()

        for aug_config in self.augmentations:
            dataframe_augmented = self._apply_augmentation(
                dataframe_augmented, aug_config
            )

        return dataframe_augmented

    def _apply_augmentation(self, dataframe, config):
        """Apply a single augmentation transformation based on configuration.

        This private method handles the application of individual augmentation
        operations as specified in the configuration dictionary.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame to augment
        config : dict
            Augmentation configuration dictionary containing 'type', 'column',
            and type-specific parameters

        Returns
        -------
        pd.DataFrame
            DataFrame with the specified augmentation applied

        Notes
        -----
        - Handles different augmentation types through conditional logic
        - Includes error handling for missing columns
        - Provides feedback on feature creation success
        """
        aug_type = config["type"]
        column = config["column"]

        if column not in dataframe.columns:
            print(f"Warning: Column '{column}' not found in dataframe")
            return dataframe

        if aug_type == "shift_log":
            # We have a plan to do a log transform on the y value. Was harder than expected so this is currently not used. TOD: Clean up?
            shift_periods = config.get("shift_periods", 1)
            shifted_values = dataframe[column].shift(shift_periods)
            log_shifted_values = np.log(shifted_values + 1e-8)
            feature_name = f"{column}_shift_{shift_periods}_log"
            dataframe[feature_name] = log_shifted_values
        elif aug_type == "shift":
            shift_periods = config.get("shift_periods", 1)
            if isinstance(shift_periods, list):
                feature_names = []
                for shift in shift_periods:
                    feature_name = self.create_shift(dataframe, shift, column)
                    feature_names.append(feature_name)
                print(f"Created augmented features: {', '.join(feature_names)}")
                return dataframe  # Return early since we've already printed
            else:
                feature_name = self.create_shift(dataframe, shift_periods, column)
        elif aug_type == "cosine":
            feature_name = f"{column}_cosine"
            dataframe[feature_name] = np.cos(
                2 * np.pi * dataframe["weekofyear_col"] / 52
            )
        elif aug_type == "sine":
            feature_name = f"{column}_sine"
            dataframe[feature_name] = np.sin(
                2 * np.pi * dataframe["weekofyear_col"] / 52
            )
        elif aug_type == "temp_composite":
            shift_periods = config.get("shift_periods", 1)
            feature_name = f"{column}_temp_composite"
            dataframe[feature_name] = (
                dataframe.groupby(level=0)["station_avg_temp_c"]
                .shift(shift_periods)
                .transform(lambda x: (x - 27.5) / 27)
            )
        elif aug_type == "saturation_composite":
            shift_periods = config.get("shift_periods", 1)
            feature_name = f"{column}_saturation_composite"

            # Calculate lagged saturation deficit per group (level=0)
            shifted_air_temp = dataframe.groupby(level=0)[
                "reanalysis_air_temp_k"
            ].shift(shift_periods)
            shifted_dew_point = dataframe.groupby(level=0)[
                "reanalysis_dew_point_temp_k"
            ].shift(shift_periods)
            dataframe[feature_name] = shifted_air_temp - shifted_dew_point
        else:
            print(f"Warning: Unknown augmentation type '{aug_type}'")
            return dataframe

        print(f"Created augmented feature: {feature_name}")
        return dataframe

    def create_shift(self, dataframe, shift_value, column):
        """Create a lagged version of a feature.

        This method creates a time-shifted version of the specified column,
        which is essential for capturing temporal dependencies in time series
        epidemiological data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame to modify (modified in-place)
        shift_value : int
            Number of periods to shift. Positive values create lags.
        column : str
            Name of the column to lag

        Returns
        -------
        str
            Name of the created lagged feature

        Notes
        -----
        - Modifies dataframe in-place
        - Lagged features help capture the delay between environmental
          conditions and disease manifestation
        - Uses pandas shift() which handles time series index alignment

        Examples
        --------
        >>> feature_name = augmenter.create_shift(df, 2, 'temperature')
        >>> print(feature_name)  # 'temperature_lag2'
        """
        shifted_values = dataframe[column].shift(shift_value)
        feature_name = f"{column}_lag{shift_value}"
        dataframe[feature_name] = shifted_values
        return feature_name
