import pandas as pd


def summarize_dataset(df: pd.DataFrame):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Column Names": list(df.columns)
    }


def check_missing_values(df: pd.DataFrame):
    return df.isnull().sum()


def get_data_types(df: pd.DataFrame):
    return df.dtypes.astype(str)


def describe_numeric(df: pd.DataFrame):
    return df.describe()


def list_columns(df: pd.DataFrame):
    return list(df.columns)

def correlation_with_target(df: pd.DataFrame, target: str):

    numeric_df = df.select_dtypes(include="number")

    if target not in numeric_df.columns:
        return f"{target} is not a numeric column."

    corr = numeric_df.corr()[target].sort_values(
        ascending=False
    )

    return corr