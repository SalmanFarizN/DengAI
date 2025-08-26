from sklearn.pipeline import Pipeline
from load_data import LoadData
from preprocess import Preprocess
from feature_selector import FeatureSelector
from feature_augmentation import FeatureAugmentation
from output_processing import OutputCSV

# from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from stats_model_wrapper import StatsModelWrapper
import statsmodels.api as sm
from datetime import datetime


def create_pipeline():
    """Create a complete machine learning pipeline for dengue fever prediction.

    This function constructs a 4-step scikit-learn pipeline optimized for
    dengue fever case count prediction using time series epidemiological data.
    The pipeline includes feature engineering, preprocessing, feature selection,
    and a gradient boosting model.

    Pipeline Steps:
    1. Feature Augmentation: Creates lagged features and seasonal transformations
    2. Preprocessing: Handles missing values using backward fill
    3. Feature Selection: Selects curated set of predictive features
    4. Model: XGBoost Regressor for final predictions

    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted pipeline ready for training on dengue fever data.
        Pipeline steps are named: 'featureaugmentation', 'preprocessing',
        'featureengineering', 'model'.

    Notes
    -----
    - Originally experimented with GLM/Negative Binomial but XGBoost performed better
    - Pipeline is designed to handle both San Juan and Iquitos data separately
    - Each city should have its own pipeline instance for optimal performance

    Examples
    --------
    >>> pipeline = create_pipeline()
    >>> pipeline.fit(X_train, y_train)
    >>> predictions = pipeline.predict(X_test)

    See Also
    --------
    main : Function that uses this pipeline for full training and prediction workflow
    """

    # glm_estimator = StatsModelWrapper(family=sm.families.NegativeBinomial(alpha=1e-08)) # WE USED THIS FOR ONE STEP BUT IN THE END DIDNT DELIVER GOOD RESULTS

    return Pipeline(
        [
            ("featureaugmentation", FeatureAugmentation()),
            ("preprocessing", Preprocess()),
            ("featureengineering", FeatureSelector()),
            ("model", XGBRegressor()),
        ]
    )


def main():
    """Execute the complete dengue fever prediction workflow.

    This function implements the full machine learning pipeline for the DengAI
    competition, including data loading, model training for both cities,
    prediction generation, and output formatting. The workflow is designed
    to handle the dual-city nature of the competition where San Juan and
    Iquitos require separate models due to different environmental conditions.

    Workflow Steps:
    1. Load training data (features + labels) for both cities
    2. Load test data (features only) for both cities
    3. Create separate pipelines for each city
    4. Train models on city-specific training data
    5. Generate predictions for test data
    6. Post-process predictions (ensure non-negative integers)
    7. Format and save predictions to timestamped CSV file

    Returns
    -------
    pd.DataFrame
        DataFrame containing San Juan predictions with columns:
        ['city', 'year', 'weekofyear', 'total_cases']

    Raises
    ------
    NotImplementedError
        If data loading fails or pipeline components are not properly implemented.
    FileNotFoundError
        If required data files are not found in expected locations.

    Notes
    -----
    - Expects data files in 'data/raw/' directory with specific naming conventions
    - Creates separate models for San Juan ('sj') and Iquitos ('iq')
    - Predictions are saved to 'data/predictions/' with timestamp
    - Uses XGBoost regressor as final model after feature engineering
    - Post-processing ensures predictions are non-negative integers

    File Dependencies
    -----------------
    - ``data/raw/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv``
    - ``data/raw/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv``
    - ``data/raw/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv``

    Examples
    --------
    >>> result = main()
    >>> print(f"Generated predictions for {len(result)} San Juan test samples")

    See Also
    --------
    create_pipeline : Creates the ML pipeline used in this function
    LoadData : Handles data loading and city separation
    OutputCSV : Manages prediction output formatting
    """
    print("main() called...")

    # Load the data
    # Training data and labels
    data_path = "data/raw/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv"
    labels_path = "data/raw/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv"

    # Test data
    test_data_path = (
        "data/raw/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv"
    )

    # Load the train data
    Sj, Iq = LoadData(data_path, labels_path).load()
    print(f"Data loaded: {len(Sj)} rows for San Juan, {len(Iq)} rows for Iquitos")

    # Extract y_train
    y_train_sj = Sj["total_cases"]
    y_train_iq = Iq["total_cases"]

    # Load the test data features
    Sj_test, Iq_test = LoadData(test_data_path).load()

    if any(x is None for x in [Sj, y_train_sj, Sj]):
        raise NotImplementedError("Data loading not implemented yet")

    pipeline_1 = create_pipeline()
    pipeline_2 = create_pipeline()

    # if any(step[1] is None for step in pipeline.steps):
    #     raise NotImplementedError("Pipeline steps not implemented yet")

    fit1 = pipeline_1.fit(Sj, y_train_sj)
    fit2 = pipeline_2.fit(Iq, y_train_iq)

    predictions_csv_1 = fit1.predict(Sj_test)
    predictions_csv_2 = fit2.predict(Iq_test)

    # TODO : Move this into a new class/method for post processing that we can add to the pipleine.
    Sj_test["total_cases"] = np.maximum(np.array(predictions_csv_1).astype(int), 0)
    Iq_test["total_cases"] = np.maximum(np.array(predictions_csv_2).astype(int), 0)

    # Append the predictions to the test data

    # Extract the city, year weekofyear and total_Cases from the test data
    predictions_csv_1 = Sj_test.reset_index()[
        ["city", "year", "weekofyear", "total_cases"]
    ]
    predictions_csv_2 = Iq_test.reset_index()[
        ["city", "year", "weekofyear", "total_cases"]
    ]

    # Save the predictions to a CSV file in 'data/predictions/' directory
    csvfile = OutputCSV(predictions_csv_1, predictions_csv_2)
    csvfile.save()

    return predictions_csv_1


if __name__ == "__main__":
    """Main execution block for dengue fever prediction script.

    Executes the main prediction workflow with error handling for
    common setup and implementation issues.
    """
    try:
        result = main()
        print(f"Successfully generated predictions. Sample shape: {result.shape}")
    except NotImplementedError as e:
        print(f"Setup needed: {e}")
    except FileNotFoundError as e:
        print(f"Required data files not found: {e}")
    except Exception as e:
        print(f"Unexpected error during execution: {e}")
        raise
