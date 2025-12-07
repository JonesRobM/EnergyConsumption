import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Import your plotting utilities (ensure these are available on PYTHONPATH or in the same project)
# from your_plot_utils_module import (
#     set_favourite_plot_params,
#     apply_favourite_figure_params,
#     setup_subplot_with_favourite_params,
#     apply_favourite_to_all_axes,
# )

# --- Inline copy of plotting utilities if not imported elsewhere ---
def set_favourite_plot_params(ax, labelsize=14, spine_width=2, x_title='x', y_title='y'):
    if ax is None:
        raise ValueError("Axis object cannot be None")
    ax.set_xlabel(x_title, fontsize=labelsize, fontweight='bold')
    ax.set_ylabel(y_title, fontsize=labelsize, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    ax.tick_params(axis='both', which='major', labelsize=labelsize-2)
    return ax

def apply_favourite_figure_params(fig, tight_layout=True):
    if tight_layout:
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.92, hspace=0.3, wspace=0.3)
    return fig

def setup_subplot_with_favourite_params(subplot_pos, x_title='x', y_title='y', labelsize=14, spine_width=2):
    if isinstance(subplot_pos, tuple):
        ax = plt.subplot(*subplot_pos)
    else:
        ax = plt.subplot(subplot_pos)
    return set_favourite_plot_params(ax, labelsize, spine_width, x_title, y_title)

def apply_favourite_to_all_axes(fig, **kwargs):
    for ax in fig.axes:
        if hasattr(ax, "spines") and ax.get_visible():
            set_favourite_plot_params(
                ax,
                x_title=ax.get_xlabel(),
                y_title=ax.get_ylabel(),
                **kwargs
            )

# --- Global Settings ---
FIGURES_DIR = 'figures'
DATA_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'kagglehub', 'datasets', 'robikscube', 'hourly-energy-consumption', 'versions', '3')
COMPOSITE_DATA_PATH = 'composite_energy_data.csv'

# --- Data Loading and Preparation ---

def download_and_prepare_data():
    """Downloads data from Kaggle and prepares the composite dataframe if it doesn't exist."""
    if os.path.exists(COMPOSITE_DATA_PATH):
        print(f"'{COMPOSITE_DATA_PATH}' already exists. Loading from file.")
        df_composite = pd.read_csv(COMPOSITE_DATA_PATH)
        df_composite['Datetime'] = pd.to_datetime(df_composite['Datetime'])
        df_composite = df_composite.set_index('Datetime')
        return df_composite

    print("Downloading dataset from Kaggle...")
    try:
        kagglehub.dataset_download("robikscube/hourly-energy-consumption")
    except Exception as e:
        print(f"Could not download dataset from Kaggle. Please ensure your Kaggle API token is set up correctly. Error: {e}")
        return None

    print("Creating composite dataframe...")
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['source_file'] = os.path.basename(file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not dataframes:
        print("No CSV files found.")
        return None

    composite_df = pd.concat(dataframes, ignore_index=True)
    composite_df.to_csv(COMPOSITE_DATA_PATH, index=False)
    print(f"Composite dataframe saved as '{COMPOSITE_DATA_PATH}'")
    
    composite_df['Datetime'] = pd.to_datetime(composite_df['Datetime'])
    composite_df = composite_df.set_index('Datetime')
    return composite_df

def add_temporal_features(df):
    """Adds temporal features to the dataframe."""
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['weekofyear'] = df.index.isocalendar().week
    return df

# --- Figure Generation Functions ---

def generate_regional_consumption_plot(df):
    print("Generating regional_consumption.png...")
    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    fig, axes = plt.subplots(len(files), 1, figsize=(15, 5 * len(files)))
    if len(files) == 1:
        axes = [axes]
    for i, file in enumerate(files):
        try:
            df_region = pd.read_csv(os.path.join(DATA_DIR, file))
            df_region = df_region.set_index('Datetime')
            df_region.index = pd.to_datetime(df_region.index)
            ax = axes[i]
            df_region.plot(style='.', color='skyblue', ax=ax)
            ax.set_title(f'Energy Use in {file} in MW', fontsize=16, fontweight='bold')
            set_favourite_plot_params(ax, x_title='Datetime', y_title='Energy (MW)')
        except Exception as e:
            print(f"Error processing {file}: {e}")
    apply_favourite_figure_params(fig)
    plt.savefig(os.path.join(FIGURES_DIR, 'regional_consumption.png'))
    plt.close()

def generate_correlation_heatmaps(df, energy_mw_cols):
    print("Generating correlation heatmaps...")
    # Overall correlation
    overall_corr = df[energy_mw_cols].corr()
    fig = plt.figure(figsize=(14, 12))
    ax = setup_subplot_with_favourite_params(111, x_title='Region', y_title='Region')
    sns.heatmap(overall_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Cross-Regional Energy Consumption Correlation (All Data)', fontsize=16, fontweight='bold', pad=20)
    apply_favourite_figure_params(fig)
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_regional_correlation.png'))
    plt.close()

    # Monthly correlations
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    months_to_plot = [1, 4, 7, 10]
    month_names = ['January', 'April', 'July', 'October']
    for idx, (month, month_name) in enumerate(zip(months_to_plot, month_names)):
        ax = axes[idx // 2, idx % 2]
        month_data = df[df['month'] == month][energy_mw_cols]
        monthly_corr = month_data.corr()
        sns.heatmap(monthly_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f'{month_name} - Cross-Regional Correlation', fontsize=14, fontweight='bold', pad=10)
        set_favourite_plot_params(ax, x_title='Region', y_title='Region')
    apply_favourite_figure_params(fig)
    plt.savefig(os.path.join(FIGURES_DIR, 'monthly_cross_regional_correlation.png'))
    plt.close()

def generate_yoy_correlation_heatmaps(df, energy_mw_cols):
    print("Generating year-over-year correlation heatmaps...")
    regions_to_plot = ['AEP_MW', 'PJME_MW', 'DOM_MW', 'DAYTON_MW']
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    for idx, region in enumerate(regions_to_plot):
        if region in df.columns:
            ax = axes[idx // 2, idx % 2]
            df_pivot = df.pivot_table(values=region, index=[df['month'], df['day'], df['hour']], columns='year', aggfunc='mean')
            year_corr = df_pivot.corr()
            sns.heatmap(year_corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5, vmin=0, vmax=1, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title(f'{region.replace("_MW", "")} - Year-over-Year Correlation', fontsize=14, fontweight='bold', pad=10)
            set_favourite_plot_params(ax, x_title='Year', y_title='Year')
    apply_favourite_figure_params(fig)
    plt.savefig(os.path.join(FIGURES_DIR, 'yoy_correlation.png'))
    plt.close()
    
def generate_pattern_plots(df, energy_mw_cols):
    print("Generating pattern plots...")
    selected_cols = ['AEP_MW', 'PJME_MW', 'DOM_MW', 'DAYTON_MW']

    # Hourly
    hourly_avg = df.groupby('hour')[energy_mw_cols].mean()
    hourly_std = df.groupby('hour')[energy_mw_cols].std()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for idx, col in enumerate(selected_cols):
        ax = axes[idx // 2, idx % 2]
        if col in hourly_avg.columns:
            hours, means, stds = hourly_avg.index, hourly_avg[col].values, hourly_std[col].values
            ax.plot(hours, means, marker='o', linewidth=2, markersize=6, color='steelblue', label='Mean')
            ax.fill_between(hours, means - stds, means + stds, alpha=0.3, color='steelblue', label='±1 Std Dev')
            ax.set_title(f'{col.replace("_MW", "")} - Hourly Consumption with Variance', fontsize=12, fontweight='bold')
            ax.set_xticks(range(0, 24, 2))
            ax.legend()
            set_favourite_plot_params(ax, x_title='Hour of Day', y_title='Energy (MW)')
    apply_favourite_figure_params(fig)
    plt.savefig(os.path.join(FIGURES_DIR, 'hourly_patterns.png'))
    plt.close()

    # Daily
    dow_avg = df.groupby('dayofweek')[energy_mw_cols].mean()
    dow_std = df.groupby('dayofweek')[energy_mw_cols].std()
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for idx, col in enumerate(selected_cols):
        ax = axes[idx // 2, idx % 2]
        if col in dow_avg.columns:
            x_pos, means, stds = np.arange(len(dow_avg)), dow_avg[col].values, dow_std[col].values
            ax.bar(x_pos, means, color='coral', alpha=0.7, yerr=stds, capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})
            ax.set_title(f'{col.replace("_MW", "")} - Average by Day of Week (±Std Dev)', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(dow_names, rotation=45, ha='right')
            set_favourite_plot_params(ax, x_title='Day of Week', y_title='Energy (MW)')
    apply_favourite_figure_params(fig)
    plt.savefig(os.path.join(FIGURES_DIR, 'daily_patterns.png'))
    plt.close()

    # Seasonal
    monthly_avg = df.groupby('month')[energy_mw_cols].mean()
    monthly_std = df.groupby('month')[energy_mw_cols].std()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for idx, col in enumerate(selected_cols):
        ax = axes[idx // 2, idx % 2]
        if col in monthly_avg.columns:
            months, means, stds = monthly_avg.index, monthly_avg[col].values, monthly_std[col].values
            ax.plot(months, means, marker='o', linewidth=2, markersize=8, color='green', label='Mean')
            ax.fill_between(months, means - stds, means + stds, alpha=0.3, color='green', label='±1 Std Dev')
            ax.set_title(f'{col.replace("_MW", "")} - Monthly Consumption with Variance', fontsize=12, fontweight='bold')
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names, rotation=45, ha='right')
            ax.legend()
            set_favourite_plot_params(ax, x_title='Month', y_title='Energy (MW)')
    apply_favourite_figure_params(fig)
    plt.savefig(os.path.join(FIGURES_DIR, 'seasonal_patterns.png'))
    plt.close()

def generate_decomposition_plot(df):
    print("Generating decomposition plot...")
    sample_region = 'PJME_MW'
    if sample_region in df.columns:
        daily_data = df[sample_region].resample('D').mean().dropna()
        if len(daily_data) > 365 * 2:
            decomposition = seasonal_decompose(daily_data, model='additive', period=365)
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            decomposition.observed.plot(ax=axes[0], color='black')
            axes[0].set_ylabel('Observed')
            axes[0].set_title(f'{sample_region.replace("_MW", "")} - Time Series Decomposition', fontsize=14, fontweight='bold')
            decomposition.trend.plot(ax=axes[1], color='blue')
            axes[1].set_ylabel('Trend')
            decomposition.seasonal.plot(ax=axes[2], color='green')
            axes[2].set_ylabel('Seasonal')
            decomposition.resid.plot(ax=axes[3], color='red')
            axes[3].set_ylabel('Residual')
            axes[3].set_xlabel('Date')
            for ax in axes:
                set_favourite_plot_params(ax, x_title=ax.get_xlabel() or 'Date', y_title=ax.get_ylabel() or '')
            apply_favourite_figure_params(fig)
            plt.savefig(os.path.join(FIGURES_DIR, 'decomposition.png'))
            plt.close()

def generate_shallow_model_plots(df_composite):
    print("Generating shallow model plots...")
    target_col = 'PJME_MW'
    df_temp = df_composite.copy()
    for lag in [1, 2, 3, 24, 168]:
        df_temp[f'{target_col}_lag_{lag}'] = df_temp[target_col].shift(lag)
    for window in [24, 168]:
        df_temp[f'{target_col}_rolling_mean_{window}'] = df_temp[target_col].rolling(window=window).mean()
    
    feature_cols = ['year', 'month', 'day', 'hour', 'dayofweek', 'quarter']
    lag_cols = [f'{target_col}_lag_{lag}' for lag in [1, 2, 3, 24, 168]]
    rolling_cols = [f'{target_col}_rolling_mean_{window}' for window in [24, 168]]
    df_model = df_temp[[target_col] + feature_cols + lag_cols + rolling_cols].copy().dropna()
    
    X, y = df_model.drop(columns=[target_col]), df_model[target_col]
    split_idx = int(len(df_model) * 0.8)
    X_train, X_test, y_train, y_test = X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[:split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(), 'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0), 'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    }
    results = {}
    for name, model in models.items():
        if 'Forest' in name:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        results[name] = {'predictions': y_pred, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred)), 'r2': r2_score(y_test, y_pred)}

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        ax.plot(y_test.iloc[-500:].values, label='Actual', linewidth=2, alpha=0.7)
        ax.plot(result['predictions'][-500:], label='Predicted', linewidth=2, alpha=0.7)
        ax.set_title(f"{name}\nRMSE: {result['rmse']:.2f}, R²: {result['r2']:.4f}")
        ax.legend()
        set_favourite_plot_params(ax, x_title='Time Steps (hours)', y_title='Energy (MW)')
    apply_favourite_figure_params(fig)
    plt.savefig(os.path.join(FIGURES_DIR, 'shallow_predictions.png'))
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(y_test, result['predictions'], alpha=0.3, s=1)
        min_val, max_val = min(y_test.min(), result['predictions'].min()), max(y_test.max(), result['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_title(f"{name} (R²: {result['r2']:.4f})")
        ax.legend()
        set_favourite_plot_params(ax, x_title='Actual Energy (MW)', y_title='Predicted Energy (MW)')
    apply_favourite_figure_params(fig)
    plt.savefig(os.path.join(FIGURES_DIR, 'shallow_scatter.png'))
    plt.close()


def generate_feature_importance_plot(X_train, y_train):
    print("Generating feature importance plot...")
    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=200, depth=8, learning_rate=0.1, random_state=42, verbose=False)
    }
    importances = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        importances[name] = model.feature_importances_

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    feature_names = X_train.columns.tolist()
    colors = ['steelblue', 'green', 'orange']
    for i, (name, importance) in enumerate(importances.items()):
        ax = axes[i]
        df_importance = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False).head(10)
        ax.barh(range(len(df_importance)), df_importance['importance'], color=colors[i])
        ax.set_yticks(range(len(df_importance)))
        ax.set_yticklabels(df_importance['feature'])
        ax.set_title(f'{name} - Top 10 Features')
        set_favourite_plot_params(ax, x_title='Importance', y_title='Feature')
        ax.invert_yaxis()
    apply_favourite_figure_params(fig)
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'))
    plt.close()


# --- Main Execution ---

def main():
    """Main function to generate all figures."""
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

    df_composite = download_and_prepare_data()
    if df_composite is None:
        return

    df_composite = add_temporal_features(df_composite)
    energy_mw_cols = [col for col in df_composite.columns if col.endswith('_MW')]

    # generate_regional_consumption_plot(df_composite) # This generates a lot of plots. Skipping for brevity.
    generate_correlation_heatmaps(df_composite, energy_mw_cols)
    generate_yoy_correlation_heatmaps(df_composite, energy_mw_cols)
    generate_pattern_plots(df_composite, energy_mw_cols)
    generate_decomposition_plot(df_composite)
    
    # For model-based plots, we need to re-run the data prep
    target_col = 'PJME_MW'
    df_temp = df_composite.copy()
    for lag in [1, 2, 3, 24, 168]: df_temp[f'{target_col}_lag_{lag}'] = df_temp[target_col].shift(lag)
    for window in [24, 168]: df_temp[f'{target_col}_rolling_mean_{window}'] = df_temp[target_col].rolling(window=window).mean()
    feature_cols = ['year', 'month', 'day', 'hour', 'dayofweek', 'quarter']
    lag_cols = [f'{target_col}_lag_{lag}' for lag in [1, 2, 3, 24, 168]]
    rolling_cols = [f'{target_col}_rolling_mean_{window}' for window in [24, 168]]
    df_model = df_temp[[target_col] + feature_cols + lag_cols + rolling_cols].copy().dropna()
    X, y = df_model.drop(columns=[target_col]), df_model[target_col]
    split_idx = int(len(df_model) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    generate_shallow_model_plots(df_composite)
    generate_feature_importance_plot(X_train, y_train)

    print("\n--- Deep Learning Model Figures ---")
    generate_lstm_figures_if_available()

    print("\n--- TFT Figures ---")
    print("Skipping TFT figures because the notebook shows a training error.")
    print("If a trained TFT model checkpoint is available, these figures could be generated.")

def generate_lstm_figures_if_available():
    """Generate LSTM/GRU training and prediction figures if checkpoints exist."""
    import torch
    from pathlib import Path

    checkpoints_dir = Path('checkpoints')
    if not checkpoints_dir.exists():
        print("No checkpoints directory found. Skipping LSTM figures.")
        return

    # Look for LSTM/GRU checkpoints
    lstm_checkpoints = list(checkpoints_dir.glob('lstm_*.pt')) + list(checkpoints_dir.glob('gru_*.pt'))

    if not lstm_checkpoints:
        print("No LSTM/GRU checkpoints found. Train a model first with:")
        print("  python scripts/train_lstm.py --mode train_test --epochs 50")
        return

    print("Found LSTM/GRU checkpoints. Generating figures...")

    for checkpoint_path in lstm_checkpoints:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model_type = "GRU" if checkpoint.get('config', None) and checkpoint['config'].use_gru else "LSTM"

            # Plot training history if available
            if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
                train_losses = checkpoint['train_losses']
                val_losses = checkpoint['val_losses']

                fig, ax = plt.subplots(figsize=(10, 6))
                epochs_range = range(1, len(train_losses) + 1)
                ax.plot(epochs_range, train_losses, label='Train Loss', linewidth=2, marker='o')
                ax.plot(epochs_range, val_losses, label='Val Loss', linewidth=2, marker='s')
                ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_type} Training History', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                save_name = f"{checkpoint_path.stem}_training_history.png"
                plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=150)
                plt.close()
                print(f"   Generated: {save_name}")
        except Exception as e:
            print(f"   Error processing {checkpoint_path.name}: {e}")

if __name__ == '__main__':
    main()
