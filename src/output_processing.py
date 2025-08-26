import pandas as pd
from datetime import datetime


class OutputCSV:
    """CSV output handler for dengue fever prediction results.
    
    This class manages the formatting and export of prediction results to CSV format
    for submission to the DengAI competition or further analysis. It handles the
    concatenation of predictions from multiple cities and ensures proper formatting
    according to competition requirements.
    
    The class automatically generates timestamped output files and handles the
    proper structure expected for dengue fever case predictions.
    
    Attributes
    ----------
    pd1 : pd.DataFrame
        Predictions DataFrame for the first city (typically San Juan)
    pd2 : pd.DataFrame
        Predictions DataFrame for the second city (typically Iquitos)
    current_time : datetime
        Timestamp of when the instance was created
    output_file_path : str
        Generated file path for the output CSV file
        
    Examples
    --------
    >>> sj_predictions = pd.DataFrame({'city': ['sj'], 'total_cases': [10]})
    >>> iq_predictions = pd.DataFrame({'city': ['iq'], 'total_cases': [5]})
    >>> output_handler = OutputCSV(sj_predictions, iq_predictions)
    >>> file_path = output_handler.save()
    """
    def __init__(self, pd1: pd.DataFrame, pd2: pd.DataFrame):
        """Initialize the CSV output handler with prediction DataFrames.
        
        Parameters
        ----------
        pd1 : pd.DataFrame
            Predictions DataFrame for the first city. Expected to contain columns:
            ['city', 'year', 'weekofyear', 'total_cases']
        pd2 : pd.DataFrame  
            Predictions DataFrame for the second city. Expected to contain columns:
            ['city', 'year', 'weekofyear', 'total_cases']
            
        Notes
        -----
        - Automatically generates a timestamped output file path
        - File path follows pattern: 'data/predictions/predictions_{timestamp}.csv'
        - Both DataFrames should have compatible structure for concatenation
        """
        self.current_time = datetime.now()
        self.output_file_path = f"data/predictions/predictions_{self.current_time}.csv"  # TODO: Dynamically add info from model type and features set to file name using Pipeline Context.
        self.pd1 = pd1
        self.pd2 = pd2

    def save(self):
        """Save the combined predictions to a CSV file.
        
        This method concatenates predictions from both cities, ensuring proper
        formatting for submission or analysis. The output follows the standard
        DengAI competition format with required columns and structure.
        
        Returns
        -------
        str
            Path to the saved CSV file
            
        Notes
        -----
        - Combines predictions from both cities using pd.concat with ignore_index=True
        - Saves with headers included for readability
        - Creates timestamped files to avoid overwriting previous predictions
        - Output directory structure: 'data/predictions/'
        
        Examples
        --------
        >>> output_handler = OutputCSV(sj_pred, iq_pred)
        >>> saved_path = output_handler.save()
        >>> print(f"Predictions saved to: {saved_path}")
        """
        # Combine predictions from both cities into a single DataFrame
        predictions_csv = pd.concat([self.pd1, self.pd2], ignore_index=True)

        # Save the combined predictions to CSV with proper formatting
        predictions_csv.to_csv(
            self.output_file_path,
            index=False,  # Don't include row indices in output
            header=True,  # Include column headers for readability
        )

        return self.output_file_path
