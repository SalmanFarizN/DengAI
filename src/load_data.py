import pandas as pd


class LoadData:
    """Data loader for DengAI disease spread prediction dataset.
    
    This class handles loading and preprocessing of DengAI training and test data,
    separating the dataset by city (San Juan and Iquitos) and optionally merging
    with target labels for supervised learning tasks.
    
    The class is designed to work with the standard DengAI competition format where
    data is provided as separate CSV files for features and labels, with a multi-level
    index structure based on city, year, and week of year.
    
    Attributes:
        data_path (str): Path to the CSV file containing feature data.
        labels_path (str, optional): Path to the CSV file containing target labels.
    """

    def __init__(self, data_path, labels_path=None):
        """Initialize the data loader with file paths.
        
        Parameters
        ----------
        data_path : str
            Path to the CSV file containing feature data. Expected to have columns
            including 'city', 'year', 'weekofyear' and various weather/environmental features.
        labels_path : str, optional
            Path to the CSV file containing target labels ('total_cases'). 
            If None, only feature data will be loaded (typical for test sets).
            Default is None.
        
        Examples
        --------
        >>> loader = LoadData('features.csv', 'labels.csv')  # For training data
        >>> loader = LoadData('test_features.csv')  # For test data without labels
        """
        self.data_path = data_path
        self.labels_path = labels_path

    def load(self):
        """Load and process data from CSV files, separating by city.
        
        This method loads the feature data, optionally merges with labels,
        creates additional index columns for downstream processing, and
        separates the data into city-specific DataFrames.
        
        Returns
        -------
        tuple of pd.DataFrame
            A tuple containing two DataFrames:
            - sj_data: DataFrame containing all data for San Juan (city='sj')
            - iq_data: DataFrame containing all data for Iquitos (city='iq')
            
            Both DataFrames have a MultiIndex with levels [city, year, weekofyear]
            and include additional columns 'city_col', 'year_col', 'weekofyear_col'
            for feature engineering purposes.
        
        Raises
        ------
        FileNotFoundError
            If the specified data_path or labels_path does not exist.
        KeyError
            If required columns ('city', 'year', 'weekofyear') are missing from the data.
        
        Examples
        --------
        >>> loader = LoadData('train_features.csv', 'train_labels.csv')
        >>> sj_train, iq_train = loader.load()
        >>> print(f"San Juan: {len(sj_train)} samples, Iquitos: {len(iq_train)} samples")
        
        Notes
        -----
        - The method preserves original index columns as separate features
        - Data is automatically filtered by city using the MultiIndex
        - If labels are provided, they are joined on the MultiIndex
        """

        # Load data and set index to city, year, weekofyear
        # NOTE: DATA CONTAINS ALL FEATURES
        df = pd.read_csv(self.data_path)

        # copy index columns so we can use them later by column name
        df["city_col"] = df["city"]
        df["year_col"] = df["year"]
        df["weekofyear_col"] = df["weekofyear"]
        df.set_index(["city", "year", "weekofyear"], inplace=True)

        if self.labels_path is not None:
            # NOTE: LABELS contains total cases AND output columns needed for final csv
            df_labels = pd.read_csv(self.labels_path)
            df_labels.set_index(["city", "year", "weekofyear"], inplace=True)
            df = df.join(df_labels)

        # Separate San Juan and Iquitos dataframes

        sj = df[df.index.get_level_values("city") == "sj"]
        iq = df[df.index.get_level_values("city") == "iq"]

        return sj, iq
