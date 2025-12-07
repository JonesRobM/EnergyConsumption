import pandas as pd

def add_temporal_features(df):
    """Adds temporal features to the dataframe."""
    df_feat = df.copy()
    df_feat['year'] = df_feat.index.year
    df_feat['month'] = df_feat.index.month
    df_feat['day'] = df_feat.index.day
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['quarter'] = df_feat.index.quarter
    df_feat['weekofyear'] = df_feat.index.isocalendar().week
    return df_feat

def create_lag_features(df, target_col, lags):
    """Creates lag features for a given target column."""
    df_lag = df.copy()
    for lag in lags:
        df_lag[f'{target_col}_lag_{lag}'] = df_lag[target_col].shift(lag)
    return df_lag

def create_rolling_features(df, target_col, windows):
    """Creates rolling mean features for a given target column."""
    df_roll = df.copy()
    for window in windows:
        df_roll[f'{target_col}_rolling_mean_{window}'] = df_roll[target_col].rolling(window=window).mean()
    return df_roll

if __name__ == '__main__':
    # Example Usage
    from data_loader import download_and_prepare_data
    
    df_composite = download_and_prepare_data()
    if df_composite is not None:
        df_with_temporal = add_temporal_features(df_composite)
        
        target = 'PJME_MW'
        df_with_lags = create_lag_features(df_with_temporal, target, lags=[1, 24, 168])
        df_full_features = create_rolling_features(df_with_lags, target, windows=[24, 168])
        
        print("Feature engineering example complete.")
        print("Columns added:", [col for col in df_full_features.columns if col not in df_composite.columns])
        print(df_full_features.head())
