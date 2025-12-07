import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

FIGURES_DIR = 'figures'

def setup_plotting():
    """Sets up seaborn styling and creates the figures directory."""
    sns.set_theme(style="whitegrid")
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

def plot_regional_consumption(data_dir):
    """Plots and saves energy consumption for each region."""
    print("Generating regional_consumption.png...")
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.csv')]
    num_files = len(files)
    fig, axes = plt.subplots(num_files, 1, figsize=(15, 5 * num_files), constrained_layout=True)
    
    for i, file in enumerate(files):
        try:
            df_region = pd.read_csv(os.path.join(data_dir, file), index_col='Datetime', parse_dates=True)
            col_name = [col for col in df_region.columns if col.endswith('_MW')][0]
            df_region[col_name].plot(style='.', color='skyblue', ax=axes[i])
            axes[i].set_title(f'Energy Use in {file.replace(".csv", "")} (MW)', fontsize=14)
            axes[i].set_ylabel('MW')
            axes[i].set_xlabel('Datetime')
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    plt.savefig(os.path.join(FIGURES_DIR, 'regional_consumption.png'))
    plt.close()

def plot_correlation_heatmaps(df, energy_mw_cols):
    """Plots and saves cross-regional and monthly correlation heatmaps."""
    print("Generating correlation heatmaps...")
    overall_corr = df[energy_mw_cols].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(overall_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Cross-Regional Energy Consumption Correlation', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_regional_correlation.png'))
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    months_to_plot = [1, 4, 7, 10]
    month_names = ['January', 'April', 'July', 'October']
    for idx, (month, month_name) in enumerate(zip(months_to_plot, month_names)):
        ax = axes[idx // 2, idx % 2]
        month_data = df[df['month'] == month][energy_mw_cols]
        sns.heatmap(month_data.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f'{month_name} - Cross-Regional Correlation', fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'monthly_cross_regional_correlation.png'))
    plt.close()

def plot_yoy_correlation_heatmaps(df, energy_mw_cols):
    """Plots and saves year-over-year correlation heatmaps."""
    print("Generating year-over-year correlation heatmaps...")
    regions_to_plot = ['AEP_MW', 'PJME_MW', 'DOM_MW', 'DAYTON_MW']
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    for idx, region in enumerate(regions_to_plot):
        if region in df.columns:
            ax = axes[idx // 2, idx % 2]
            df_pivot = df.pivot_table(values=region, index=[df.index.month, df.index.day, df.index.hour], columns=df.index.year)
            sns.heatmap(df_pivot.corr(), annot=True, fmt='.2f', cmap='RdYlGn', center=0.5, vmin=0, vmax=1, square=True, ax=ax)
            ax.set_title(f'{region.replace("_MW", "")} - Year-over-Year Correlation', fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'yoy_correlation.png'))
    plt.close()

def plot_consumption_patterns(df, energy_mw_cols):
    """Plots and saves hourly, daily, and seasonal consumption patterns."""
    print("Generating pattern plots...")
    selected_cols = ['AEP_MW', 'PJME_MW', 'DOM_MW', 'DAYTON_MW']

    # Hourly
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for idx, col in enumerate(selected_cols):
        ax = axes[idx // 2, idx % 2]
        if col in df.columns:
            hourly_avg = df.groupby(df.index.hour)[col].mean()
            hourly_std = df.groupby(df.index.hour)[col].std()
            ax.plot(hourly_avg.index, hourly_avg, marker='o', color='steelblue', label='Mean')
            ax.fill_between(hourly_avg.index, hourly_avg - hourly_std, hourly_avg + hourly_std, alpha=0.3, color='steelblue', label='±1 Std Dev')
            ax.set_title(f'{col.replace("_MW", "")} - Hourly Consumption'), ax.set_xlabel('Hour of Day'), ax.set_ylabel('MW'), ax.legend(), ax.grid(True)
    plt.tight_layout(), plt.savefig(os.path.join(FIGURES_DIR, 'hourly_patterns.png')), plt.close()

    # Daily
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for idx, col in enumerate(selected_cols):
        ax = axes[idx // 2, idx % 2]
        if col in df.columns:
            dow_avg = df.groupby(df.index.dayofweek)[col].mean()
            dow_std = df.groupby(df.index.dayofweek)[col].std()
            ax.bar(dow_avg.index, dow_avg, color='coral', yerr=dow_std, capsize=5)
            ax.set_title(f'{col.replace("_MW", "")} - Daily Consumption'), ax.set_xlabel('Day of Week'), ax.set_ylabel('MW'), ax.set_xticks(range(7)), ax.set_xticklabels(dow_names), ax.grid(True)
    plt.tight_layout(), plt.savefig(os.path.join(FIGURES_DIR, 'daily_patterns.png')), plt.close()

    # Seasonal
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for idx, col in enumerate(selected_cols):
        ax = axes[idx // 2, idx % 2]
        if col in df.columns:
            monthly_avg = df.groupby(df.index.month)[col].mean()
            monthly_std = df.groupby(df.index.month)[col].std()
            ax.plot(monthly_avg.index, monthly_avg, marker='o', color='green', label='Mean')
            ax.fill_between(monthly_avg.index, monthly_avg - monthly_std, monthly_avg + monthly_std, alpha=0.3, color='green', label='±1 Std Dev')
            ax.set_title(f'{col.replace("_MW", "")} - Monthly Consumption'), ax.set_xlabel('Month'), ax.set_ylabel('MW'), ax.set_xticks(range(1, 13)), ax.set_xticklabels(month_names), ax.legend(), ax.grid(True)
    plt.tight_layout(), plt.savefig(os.path.join(FIGURES_DIR, 'seasonal_patterns.png')), plt.close()

def plot_decomposition(df):
    """Plots and saves time series decomposition."""
    print("Generating decomposition plot...")
    sample_region = 'PJME_MW'
    if sample_region in df.columns:
        daily_data = df[sample_region].resample('D').mean().dropna()
        if len(daily_data) > 365 * 2:
            decomposition = seasonal_decompose(daily_data, model='additive', period=365)
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            decomposition.observed.plot(ax=axes[0], color='black'), axes[0].set_ylabel('Observed')
            decomposition.trend.plot(ax=axes[1], color='blue'), axes[1].set_ylabel('Trend')
            decomposition.seasonal.plot(ax=axes[2], color='green'), axes[2].set_ylabel('Seasonal')
            decomposition.resid.plot(ax=axes[3], color='red'), axes[3].set_ylabel('Residual')
            plt.tight_layout(), plt.savefig(os.path.join(FIGURES_DIR, 'decomposition.png')), plt.close()

def plot_model_predictions(results, y_test):
    """Plots and saves model predictions vs actuals."""
    print("Generating model prediction plots...")
    num_models = len(results)
    fig, axes = plt.subplots(num_models, 1, figsize=(15, 5 * num_models), constrained_layout=True)
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        ax.plot(y_test.iloc[-500:].values, label='Actual', alpha=0.7)
        ax.plot(result['predictions'][-500:], label='Predicted', alpha=0.7)
        ax.set_title(f"{name} - Predictions vs Actual"), ax.legend(), ax.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, 'shallow_predictions.png'))
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(y_test, result['predictions'], alpha=0.3, s=1)
        min_val, max_val = min(y_test.min(), result['predictions'].min()), max(y_test.max(), result['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_title(f"{name}"), ax.set_xlabel('Actual'), ax.set_ylabel('Predicted'), ax.legend(), ax.grid(True)
    plt.tight_layout(), plt.savefig(os.path.join(FIGURES_DIR, 'shallow_scatter.png')), plt.close()


def plot_feature_importance(importances, X_train):
    """Plots and saves feature importances."""
    print("Generating feature importance plot...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    feature_names = X_train.columns.tolist()
    for i, (name, importance) in enumerate(importances.items()):
        df_importance = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False).head(10)
        axes[i].barh(range(len(df_importance)), df_importance['importance'])
        axes[i].set_yticks(range(len(df_importance))), axes[i].set_yticklabels(df_importance['feature']), axes[i].set_title(f'{name} - Top 10 Features')
    plt.tight_layout(), plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png')), plt.close()

if __name__ == '__main__':
    # This is for testing the plotting functions independently
    # You would need to load the data first
    print("This script is intended to be used as a module.")
