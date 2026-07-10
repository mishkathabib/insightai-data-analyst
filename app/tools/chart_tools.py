import matplotlib.pyplot as plt
import pandas as pd


def generate_histogram(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots()
    ax.hist(df[column].dropna(), bins=20)
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    return fig