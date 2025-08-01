import pandas as pd
from datetime import datetime


class OutputCSV:
    def __init__(self, pd1: pd.DataFrame, pd2: pd.DataFrame):
        self.current_time = datetime.now()
        self.output_file_path = f"data/predictions/predictions_{self.current_time}.csv"
        self.pd1 = pd1
        self.pd2 = pd2

    def save(self):
        # Append the predictions from both cities
        predictions_csv = pd.concat([self.pd1, self.pd2], ignore_index=True)

        # Save the predictions to a CSV file using pandas
        predictions_csv.to_csv(
            self.output_file_path,
            index=False,
            header=True,
        )

        return self.output_file_path
