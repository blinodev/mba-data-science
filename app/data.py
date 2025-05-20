import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def check_columns(df, expected_cols):
    return set(df.columns) == set(expected_cols)
