import pandas as pd

def dst_func_indicador(df):
    df_copy = df.copy()
    df_copy["SMA 15"] = df_copy["dst"].rolling(15).mean().shift(1)
    df_copy["SMA 60"] = df_copy["dst"].rolling(60).mean().shift(1)
    df_copy["MSD 15"] = df_copy["prox_h"].rolling(15).std().shift(1)
    df_copy["MSD 60"] = df_copy["prox_h"].rolling(60).std().shift(1)
    df_copy["Min 15"] = df_copy["dst"].rolling(15).min().shift(1)
    df_copy["Max 15"] = df_copy["dst"].rolling(15).max().shift(1)
    df_copy["Median 15"] = df_copy["dst"].rolling(15).median().shift(1)
    df_copy["Taxa 15"] = df_copy["dst"] - df_copy["dst"].shift(15)
    df_copy["vel_mag"] = df_copy["vel"] * df_copy["mag"]
    return df_copy.dropna()
