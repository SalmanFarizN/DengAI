import os
import glob
import pandas as pd
from main import main

def test_csv_output():
    """Test that main() creates a CSV with the correct headers."""
    expected_columns = ['city', 'year', 'weekofyear', 'total_cases']

    # Clean up any existing prediction files before test
    existing_files = glob.glob("data/predictions/predictions_*.csv")
    for file in existing_files:
        os.remove(file)

    # Run main function
    main()

    # Check that a predictions file was created
    prediction_files = glob.glob("data/predictions/predictions_*.csv")
    assert len(prediction_files) > 0, "No predictions CSV file was created"

    # Get the created file (assuming only one after cleanup)
    output_path = prediction_files[0]

    # Verify file exists and has correct columns
    assert os.path.exists(output_path), f"CSV file {output_path} was not created"
    df = pd.read_csv(output_path)
    assert list(df.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(df.columns)}"

    # Optional: Clean up after test
    os.remove(output_path)