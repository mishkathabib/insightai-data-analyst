import sqlite3
import pandas as pd


def load_dataframe_to_sqlite(df: pd.DataFrame, table_name: str = "uploaded_data"):
    conn = sqlite3.connect(":memory:")
    df.to_sql(table_name, conn, index=False, if_exists="replace")
    return conn


def run_sql_query(df: pd.DataFrame, query: str):
    conn = load_dataframe_to_sqlite(df)

    try:
        result = pd.read_sql_query(query, conn)
        return result
    except Exception as e:
        return f"SQL error: {e}"
    finally:
        conn.close()