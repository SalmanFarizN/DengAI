from pandas.conftest import datapath
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class FeatureAugmentation(BaseEstimator, TransformerMixin):
    """Augment features by creating multiple types of derived features.
    This class implements the scikit-learn BaseEstimator and TransformerMixin interfaces.
    It can be used in a scikit-learn pipeline for feature augmentation.
    """

    def __init__(self, augmentations=None):
        """
        Args:
            augmentations (list): List of augmentation dictionaries. Each dict should contain:
                - 'type': Type of augmentation ('shift_log', 'shift')
                - 'column': Column name to apply augmentation to
                - Additional parameters specific to each augmentation type
        """
        if augmentations is None:
            # Default augmentation for backward compatibility
            augmentations = [
                                {'type': 'shift', 'column': 'reanalysis_specific_humidity_g_per_kg', 'shift_periods': [1,2,3,4]},
                                {'type': 'shift', 'column': 'reanalysis_dew_point_temp_k', 'shift_periods': [1,2,3,4]},
                                {'type': 'shift', 'column': 'reanalysis_min_air_temp_k', 'shift_periods': [1,2,3,4]},
                                {'type': 'shift', 'column': 'station_min_temp_c', 'shift_periods': [1,2,3,4]},
                                {'type': 'shift', 'column': 'reanalysis_relative_humidity_percent', 'shift_periods': [1,2,3,4]},
                                {'type': 'shift', 'column': 'station_avg_temp_c', 'shift_periods': [1,2,3,4]},
                                {'type': 'shift', 'column': 'reanalysis_precip_amt_kg_per_m2', 'shift_periods': [1,2,3,4]},
                                {'type': 'shift', 'column': 'reanalysis_air_temp_k', 'shift_periods': [1,2,3,4]},
                                {'type': 'shift', 'column': 'reanalysis_sat_precip_amt_mm', 'shift_periods': [1,2,3,4]},
                                {'type': 'shift', 'column': 'reanalysis_avg_temp_k', 'shift_periods': [1,2,3,4]},
                                {'type': 'shift', 'column': 'station_max_temp_c', 'shift_periods': [1,2,3,4]},
                                {'type': 'shift', 'column': 'station_precip_mm', 'shift_periods': [1,2,3,4]},
                                {'type': 'shift', 'column': 'ndvi_sw', 'shift_periods': [1,2,3,4]},
                                {'type': 'cosine', 'column': 'weekofyear_col', 'shift_periods': 0},
                                {'type': 'sine', 'column': 'weekofyear_col', 'shift_periods': 0},

                                # {'type': 'temp_composite', 'column': 'station_avg_temp_c', 'shift_periods': 3},
                                # {'type': 'saturation_composite', 'column': 'reanalysis_air_temp_k', 'shift_periods': 2}
                            ]

        self.augmentations = augmentations

    def fit(self, X=None, y=None):
        # No fitting necessary, so we return self
        return self

    def transform(self, dataframe):
        """Create augmented features from the dataframe based on configuration.
        Args:
            dataframe (pd.DataFrame): Input dataframe.
        Returns:
            pd.DataFrame: Dataframe with augmented features added.
        """
        # Make a copy to avoid modifying the original
        dataframe_augmented = dataframe.copy()

        for aug_config in self.augmentations:
            dataframe_augmented = self._apply_augmentation(dataframe_augmented, aug_config)

        return dataframe_augmented

    def _apply_augmentation(self, dataframe, config):
        """Apply a single augmentation based on configuration."""
        aug_type = config['type']
        column = config['column']

        if column not in dataframe.columns:
            print(f"Warning: Column '{column}' not found in dataframe")
            return dataframe

        if aug_type == 'shift_log':
            shift_periods = config.get('shift_periods', 1)
            shifted_values = dataframe[column].shift(shift_periods)
            log_shifted_values = np.log(shifted_values + 1e-8)
            feature_name = f"{column}_shift_{shift_periods}_log"
            dataframe[feature_name] = log_shifted_values
        elif aug_type == 'shift':
            shift_periods = config.get('shift_periods', 1)
            if isinstance(shift_periods, list):
                feature_names = []
                for shift in shift_periods:
                    feature_name = self.create_shift(dataframe, shift, column)
                    feature_names.append(feature_name)
                print(f"Created augmented features: {', '.join(feature_names)}")
                return dataframe  # Return early since we've already printed
            else:
                feature_name = self.create_shift(dataframe, shift_periods, column)
        elif aug_type == 'cosine':
            feature_name = f"{column}_cosine"
            dataframe[feature_name] = np.cos(2 * np.pi * dataframe["weekofyear_col"] / 52)
        elif aug_type == 'sine':
            feature_name = f"{column}_sine"
            dataframe[feature_name] = np.sin(2 * np.pi * dataframe["weekofyear_col"] / 52)
        elif aug_type == 'temp_composite':
            shift_periods = config.get('shift_periods', 1)
            feature_name = f"{column}_temp_composite"
            dataframe[feature_name] = dataframe.groupby(level=0)['station_avg_temp_c'].shift(shift_periods).transform(lambda x: (x - 27.5) / 27)
        elif aug_type == 'saturation_composite':
            shift_periods = config.get('shift_periods', 1)
            feature_name = f"{column}_saturation_composite"

            # Calculate lagged saturation deficit per group (level=0)
            shifted_air_temp = dataframe.groupby(level=0)['reanalysis_air_temp_k'].shift(shift_periods)
            shifted_dew_point = dataframe.groupby(level=0)['reanalysis_dew_point_temp_k'].shift(shift_periods)
            dataframe[feature_name] = shifted_air_temp - shifted_dew_point
        else:
            print(f"Warning: Unknown augmentation type '{aug_type}'")
            return dataframe

        print(f"Created augmented feature: {feature_name}")
        return dataframe

    def create_shift(self, dataframe, shift_value, column):
        shifted_values = dataframe[column].shift(shift_value)
        feature_name = f"{column}_lag{shift_value}"
        dataframe[feature_name] = shifted_values
        return feature_name


