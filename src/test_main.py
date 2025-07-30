import os
import pandas as pd
from main import main

def test_csv_output():
    """Simple test: does main() create a CSV with the right columns?"""

    expected_columns = ['city', 'year', 'weekofyear', 'total_cases']

    try:
        main()

        # Find any CSV files
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

        if not csv_files:
            print("❌ No CSV files found")
            return

        # Check first CSV file
        df = pd.read_csv(csv_files[0])

        if list(df.columns) == expected_columns:
            print(f"✅ CSV created with correct columns: {csv_files[0]}")
            print(f"   Rows: {len(df)}")
        else:
            print(f"❌ Wrong columns. Expected: {expected_columns}")
            print(f"   Got: {list(df.columns)}")

    except NotImplementedError:
        print("⏳ Pipeline not implemented yet")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_csv_output()