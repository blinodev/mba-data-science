def clean_data(df):
    return df.dropna()

def engineer_features(df):
    df['feature_sq'] = df['feature'] ** 2
    return df
