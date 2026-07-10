import pandas as pd


def preprocess_data(df: pd.DataFrame, target: str):
    """
    Prepares a dataset for machine learning.

    Returns:
        X = processed features
        y = target column
    """

    data = df.copy()

    # Remove obvious identifier columns
    columns_to_drop = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]

    existing = [
        col for col in columns_to_drop
        if col in data.columns
    ]

    data = data.drop(columns=existing)

    # Fill missing numeric values
    numeric_cols = data.select_dtypes(include="number").columns

    for col in numeric_cols:
        data[col] = data[col].fillna(
            data[col].median()
        )

    # Fill missing categorical values
    categorical_cols = data.select_dtypes(include="object").columns

    for col in categorical_cols:
        data[col] = data[col].fillna(
            data[col].mode()[0]
        )

    y = data[target]

    X = data.drop(columns=[target])

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    return X, y