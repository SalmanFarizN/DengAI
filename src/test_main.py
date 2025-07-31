import os
import pandas as pd
from main import main

def test_csv_output():
    """Test that main() creates a CSV with the correct headers."""

    expected_columns = ['city', 'year', 'weekofyear', 'total_cases']
    output_path = "predictions.csv"

    # Clean up any existing file before test
    if os.path.exists(output_path):
        os.remove(output_path)

    main()

    assert os.path.exists(output_path), "CSV file was not created"

    df = pd.read_csv(output_path)
    assert list(df.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(df.columns)}"
