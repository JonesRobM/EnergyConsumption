import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.feature_engineering import create_lag_features, create_rolling_features

def prepare_modelling_data(df_composite, target_col='PJME_MW'):
    """Prepares the data for modelling by creating features and splitting."""
    
    df_temp = df_composite.copy()
    df_temp = create_lag_features(df_temp, target_col, lags=[1, 2, 3, 24, 168])
    df_temp = create_rolling_features(df_temp, target_col, windows=[24, 168])

    feature_cols = ['year', 'month', 'day', 'hour', 'dayofweek', 'quarter']
    lag_cols = [f'{target_col}_lag_{lag}' for lag in [1, 2, 3, 24, 168]]
    rolling_cols = [f'{target_col}_rolling_mean_{window}' for window in [24, 168]]
    
    all_cols = [target_col] + feature_cols + lag_cols + rolling_cols
    df_model = df_temp[all_cols].copy().dropna()

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    split_idx = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def train_shallow_models(X_train, y_train, X_test, X_train_scaled, X_test_scaled):
    """Trains and evaluates shallow learning models."""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        if 'Forest' in name:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    return results

def train_gradient_boosting_models(X_train, y_train, X_test, y_test):
    """Trains and evaluates gradient boosting models."""
    gb_models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=200, depth=8, learning_rate=0.1, random_state=42, verbose=False)
    }
    
    gb_results = {}
    for name, model in gb_models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        gb_results[name] = {
            'model': model,
            'predictions': y_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    return gb_results

if __name__ == '__main__':
    from data_loader import download_and_prepare_data
    from feature_engineering import add_temporal_features

    df = download_and_prepare_data()
    if df is not None:
        df = add_temporal_features(df)
        X_train, X_test, y_train, y_test, _, _ = prepare_modelling_data(df)
        
        shallow_results = train_shallow_models(X_train, y_train, X_test, X_train, X_test) # Using non-scaled for RF
        gb_results = train_gradient_boosting_models(X_train, y_train, X_test, y_test)
        
        print("\n--- Shallow Model Results ---")
        for name, res in shallow_results.items():
            print(f"{name}: RMSE={res['rmse']:.2f}, MAE={res['mae']:.2f}, R²={res['r2']:.4f}")
            
        print("\n--- Gradient Boosting Model Results ---")
        for name, res in gb_results.items():
            print(f"{name}: RMSE={res['rmse']:.2f}, MAE={res['mae']:.2f}, R²={res['r2']:.4f}")
