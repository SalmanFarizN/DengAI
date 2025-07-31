from sklearn.pipeline import Pipeline
from src.csv_saver import CSVSaver
from src.load_data import LoadData
from src.preprocess import Preprocess
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from xgboost import XGBRegressor


def create_pipeline():
    """Creates a 3 step pipline"""
    return Pipeline(
        [
            ("preprocessing", Preprocess()),
            (
                "model",
                XGBRegressor(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=0,
                ),
            ),
            # (
            #     "save",
            #     CSVSaver(),
            # ),  # TODO: Update CSV saver it just return an empty csv now
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
    Sj, Iq, sj_labels, iq_labels = LoadData(data_path, labels_path).load()
    print(f"Data loaded: {len(Sj)} rows for San Juan, {len(Iq)} rows for Iquitos")
    # Extract y_train
    y_train_sj = sj_labels["total_cases"]
    y_train_iq = iq_labels["total_cases"]

    # Load the test data features
    Sj_test, Iq_test, _, _ = LoadData(test_data_path).load()

    if any(x is None for x in [Sj, y_train_sj, Sj]):
        raise NotImplementedError("Data loading not implemented yet")

    pipeline = create_pipeline()

    if any(step[1] is None for step in pipeline.steps):
        raise NotImplementedError("Pipeline steps not implemented yet")

    fit1 = pipeline.fit(Sj, y_train_sj)
    fit2 = pipeline.fit(Iq, y_train_iq)

    predictions_csv_1 = fit1.predict(Sj_test)
    predictions_csv_2 = fit2.predict(Iq_test)

    Sj_test["total_cases"] = np.round(predictions_csv_1, 0).astype(int)
    Iq_test["total_cases"] = np.round(predictions_csv_2, 0).astype(int)

    # Append the predictions to the test data

    # Extract the city, year weekofyear and total_Cases from the test data
    predictions_csv_1 = Sj_test.reset_index()[
        ["city", "year", "weekofyear", "total_cases"]
    ]
    predictions_csv_2 = Iq_test.reset_index()[
        ["city", "year", "weekofyear", "total_cases"]
    ]

    # Appemnd the predictions from both cities
    predictions_csv_1 = pd.concat(
        [predictions_csv_1, predictions_csv_2], ignore_index=True
    )

    # Save the predictions to a CSV file using pandas
    predictions_csv_1.to_csv(
        "data/predictions.csv", index=False, header=True, float_format="%.2f"
    )

    return predictions_csv_1


if __name__ == "__main__":
    try:
        result = main()
    except NotImplementedError as e:
        print(f"Setup needed: {e}")
