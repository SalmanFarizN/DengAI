from sklearn.pipeline import Pipeline
from csv_saver import CSVSaver

def create_pipeline():
    """Creates a 3 step pipline """
    return Pipeline([
        ('preprocessing', 'passthrough'),  # TODO: Add custom preprocessor
        ('model', 'passthrough'),         # TODO: Add custom model wrapper
        ('save', CSVSaver())          # TODO: Update CSV saver it just return an empty csv now
    ])

def main():
    print("main() called...")

    # TODO: load CSVs and assign to these variables

    X_train = [[1, 2], [3, 4]]
    y_train = [0, 1]
    X_test = [[5, 6], [7, 8]]

    if any(x is None for x in [X_train, y_train, X_test]):
        raise NotImplementedError("Data loading not implemented yet")

    pipeline = create_pipeline()

    if any(step[1] is None for step in pipeline.steps):
        raise NotImplementedError("Pipeline steps not implemented yet")

    pipeline.fit(X_train, y_train)
    predictions_csv = pipeline.transform(X_test)

    return predictions_csv

if __name__ == "__main__":
    try:
        result = main()
    except NotImplementedError as e:
        print(f"Setup needed: {e}")