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
    """Creates a 3 step pipline"""

    glm_estimator = StatsModelWrapper(family=sm.families.NegativeBinomial(alpha=1e-08))

    return Pipeline(
        [
            ("featureaugmentation", FeatureAugmentation()),
            ("preprocessing", Preprocess()),
            ("featureengineering", FeatureSelector()),
            ("model", XGBRegressor()),
        ]
    )

def main():
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


    #TODO : Move this into a new class/method for post processing that we can add to the pipleine.
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
    try:
        result = main()
    except NotImplementedError as e:
        print(f"Setup needed: {e}")
