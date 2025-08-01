import pandas as pd


class LoadData:
    """Load data from CSV file and return dataframes for San Juan and Iquitos separately.
    The labels are appended if a labels path is provided.
    """

    def __init__(self, data_path, labels_path=None):
        """Assign the path to the data and labels
        Args:
            data_path (str): Path to the CSV file containing the data.
            labels_path (str, optional): Path to the CSV file containing labels. Defaults to None.
        Returns:
            None
        """
        self.data_path = data_path
        self.labels_path = labels_path

    def load(self):
        """Load data from the specified CSV file and return separate dataframes for San Juan and Iquitos.
        Args:
            None
        Returns:
            tuple: Two pandas DataFrames, one for San Juan and one for Iquitos.
        """

        # Load data and set index to city, year, weekofyear
        # NOTE: DATA CONTAINS ALL FEATURES
        df = pd.read_csv(self.data_path)

        # copy index columns so we can use them later by column name
        df['city_col'] = df['city']
        df['year_col'] = df['year']
        df['weekofyear_col'] = df['weekofyear']
        df.set_index(['city', 'year', 'weekofyear'], inplace=True)

        if self.labels_path is not None:
            # NOTE: LABELS contains total cases AND output columns needed for final csv
            df_labels = pd.read_csv(self.labels_path)
            df_labels.set_index(['city', 'year', 'weekofyear'], inplace=True)
            df = df.join(df_labels)


        # Separate San Juan and Iquitos dataframes

        sj = df[df.index.get_level_values("city") == "sj"]
        iq = df[df.index.get_level_values("city") == "iq"]

        return sj, iq
