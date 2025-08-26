import os
import glob
import pandas as pd
from main import main


def test_csv_output():
    """Test the end-to-end prediction workflow and CSV output formatting.
    
    This integration test validates that the main prediction pipeline:
    1. Successfully executes without errors
    2. Creates a timestamped CSV output file
    3. Produces output with correct column structure for competition submission
    4. Generates predictions for both cities (San Juan and Iquitos)
    
    The test performs cleanup before and after execution to ensure a clean
    testing environment and validates the essential requirements for the
    DengAI competition submission format.
    
    Raises
    ------
    AssertionError
        If the main pipeline fails to create output files or if the output
        format doesn't match expected competition requirements.
    FileNotFoundError
        If required input data files are missing.
        
    Notes
    -----
    - Cleans up existing prediction files before test execution
    - Validates presence of required columns: ['city', 'year', 'weekofyear', 'total_cases']
    - Removes generated files after validation to maintain clean state
    - Tests the complete workflow from data loading to CSV export
    
    Examples
    --------
    >>> test_csv_output()  # Should complete without assertions if successful
    
    See Also
    --------
    main : The main prediction function being tested
    OutputCSV : CSV output handler that formats the final results
    """
    # Define expected column structure for DengAI competition submission
    expected_columns = ["city", "year", "weekofyear", "total_cases"]

    # Clean up any existing prediction files to ensure clean test environment
    existing_files = glob.glob("data/predictions/predictions_*.csv")
    for file in existing_files:
        os.remove(file)

    # Execute the main prediction pipeline
    main()

    # Verify that a predictions file was successfully created
    prediction_files = glob.glob("data/predictions/predictions_*.csv")
    assert len(prediction_files) > 0, "No predictions CSV file was created"

    # Get the most recently created prediction file
    output_path = prediction_files[0]

    # Validate file existence and structure
    assert os.path.exists(output_path), f"CSV file {output_path} was not created"
    
    # Load and validate CSV structure
    df = pd.read_csv(output_path)
    assert (
        list(df.columns) == expected_columns
    ), f"Expected columns {expected_columns}, got {list(df.columns)}"
    
    # Additional validation: Check that we have predictions for both cities
    cities = df['city'].unique()
    expected_cities = ['sj', 'iq']
    assert set(cities) == set(expected_cities), f"Expected cities {expected_cities}, got {list(cities)}"
    
    # Validate that predictions are non-negative integers (competition requirement)
    assert all(df['total_cases'] >= 0), "All predictions must be non-negative"
    assert all(df['total_cases'] == df['total_cases'].astype(int)), "All predictions must be integers"

    # Clean up after successful test
    os.remove(output_path)