import pandas as pd

class CSVSaver:
    def __init__(self, output_path="./predictions.csv"):
        self.output_path = output_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create mock CSV with correct headers
        pd.DataFrame(columns=['city', 'year', 'weekofyear', 'total_cases']).to_csv(self.output_path, index=False)
        print(f"CSV saved: {self.output_path}")
        return X # We return X for convention incase we want to chain more steps