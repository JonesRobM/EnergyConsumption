# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EnergyConsumption is a comprehensive time series forecasting project that analyzes hourly energy consumption data from multiple US regions (Kaggle dataset). The project implements a complete ML pipeline from exploratory analysis through state-of-the-art deep learning forecasting using Temporal Fusion Transformers (TFT).

**Dataset**: Hourly Energy Consumption (145K+ samples, 2002-2018)
- Primary region analyzed: PJME (Pennsylvania-New Jersey-Maryland Interconnection)
- 12 energy regions total with varying data coverage
- Target: Multi-horizon energy consumption forecasting (24-hour predictions)

## Development Environment

- **Python Environment**: A virtual environment is located at `.venv/`
- **Platform**: Windows (win32)
- **Python Version**: 3.9+

## Setup and Installation

### 1. Activate the virtual environment
```bash
.venv\Scripts\activate
```

### 2. Install dependencies

Using pip with requirements.txt:
```bash
pip install -r requirements.txt
```

Or using pip with pyproject.toml:
```bash
pip install -e .
```

For development tools (pytest, black, ruff, mypy):
```bash
pip install -e ".[dev]"
```

### 3. Configure Kaggle API

To download datasets from Kaggle, you need to set up authentication:

1. Create a Kaggle account and go to https://www.kaggle.com/account
2. Scroll to "API" section and click "Create New API Token"
3. This downloads `kaggle.json` - place it in `~/.kaggle/` (or `C:\Users\<username>\.kaggle\` on Windows)
4. Alternatively, set environment variables:
   ```bash
   set KAGGLE_USERNAME=your_username
   set KAGGLE_KEY=your_api_key
   ```

## Common Commands

### Open the Main Analysis Notebook
```bash
jupyter notebook exploration.ipynb
```

The `exploration.ipynb` notebook contains the entire analysis pipeline and should be run sequentially.

### Download the Kaggle Dataset (already integrated in notebook)
```bash
kaggle datasets download -d robikscube/hourly-energy-consumption
```

### Linting and Formatting (if dev dependencies installed)
```bash
black . --line-length 100
ruff check .
mypy .
```

### Run tests (if tests are added)
```bash
pytest
```

## Technology Stack

### Data Science & Visualization
- **NumPy** (>=1.24.0): Numerical computing and array operations
- **Pandas** (>=2.0.0): Data manipulation and time series analysis
- **SciPy** (>=1.10.0): Statistical functions and tests
- **Matplotlib/Seaborn/Plotly**: Comprehensive visualization suite

### Machine Learning (Shallow Learning)
- **scikit-learn** (>=1.3.0): Traditional ML (Linear, Ridge, Lasso, Random Forest)
- **XGBoost** (>=2.0.0): Gradient boosting with regularization
- **LightGBM** (>=4.0.0): Fast gradient boosting
- **CatBoost** (>=1.2.0): Gradient boosting with categorical support
- **statsmodels** (>=0.14.0): Time series decomposition, statistical tests

### Deep Learning & Time Series Forecasting
- **PyTorch** (>=2.1.0): Core deep learning framework
- **PyTorch Lightning** (>=2.1.0): Training framework with callbacks, logging, early stopping
- **PyTorch Forecasting** (>=1.0.0): **Main library for Temporal Fusion Transformer**
- **TensorFlow** (>=2.15.0): Available for alternative implementations

### Utilities
- **Jupyter**: Interactive notebook environment
- **Kaggle API**: Automated dataset downloads via kagglehub
- **tqdm**: Progress bars for training loops
- **python-dotenv**: Environment variable management

## Project Structure & Architecture

### Current File Structure
```
EnergyConsumption/
├── exploration.ipynb           # Main notebook (COMPLETE PIPELINE)
├── composite_energy_data.csv   # Processed data (auto-generated)
├── requirements.txt            # Dependencies
├── pyproject.toml              # Project config
├── CLAUDE.md                   # This file
├── README.md                   # User documentation
├── .venv/                      # Virtual environment
└── lightning_logs/             # TFT training logs (auto-generated)
```

### exploration.ipynb Structure

The notebook is organized into sequential sections - **run cells in order**:

#### 1. Data Loading & Preprocessing
- Downloads dataset from Kaggle using kagglehub
- Loads 13 CSV files (different energy regions)
- Creates composite dataframe with 1.2M+ rows
- Extracts temporal features: year, month, day, hour, dayofweek, quarter, weekofyear

#### 2. Exploratory Data Analysis (EDA)
- **Correlation Analysis**:
  - Overall cross-regional correlation heatmaps
  - Month-by-month correlation (seasonal patterns)
  - Year-over-year correlation (temporal consistency)
- **Statistical Analysis**:
  - Distribution tests (Anderson-Darling, skewness, kurtosis)
  - Hourly consumption patterns **with variance** (±1 std dev bands)
  - Day of week patterns **with error bars**
  - Seasonal/monthly patterns **with variance**
  - Weekend vs weekday t-tests
  - Time series decomposition (trend, seasonal, residual)

#### 3. Shallow Learning Models
- **Data Preparation**:
  - Lag features: 1hr, 2hr, 3hr, 24hr, 168hr (1 week)
  - Rolling statistics: 24hr mean, 168hr mean
  - Train/test split: 80/20 with temporal ordering preserved
  - StandardScaler for feature normalization
- **Models Implemented**:
  - Linear Regression (baseline)
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)
  - Random Forest (100 trees, max_depth=20)
  - XGBoost (200 estimators, max_depth=8)
  - LightGBM (200 estimators, max_depth=8)
  - CatBoost (200 iterations, depth=8)
- **Evaluation**:
  - Metrics: RMSE, MAE, R² on test set
  - Prediction visualizations (time series plots, scatter plots)
  - Feature importance analysis (for tree-based models)
  - Model comparison table

#### 4. Temporal Fusion Transformer (TFT)
This is the **main deep learning implementation** for production forecasting.

- **Data Preparation for TFT**:
  - Creates time_idx (sequential hour counter)
  - Cyclical features: sin/cos encodings for hour, month, dayofweek
  - Lag features: 24hr, 168hr
  - Rolling features: 24hr mean, 168hr mean
  - TimeSeriesDataSet configuration:
    - Encoder length: 168 hours (7 days lookback)
    - Prediction length: 24 hours (1 day forecast)
    - Train/Val/Test: 70/15/15 split

- **Model Architecture**:
  - Hidden size: 64
  - Attention heads: 4
  - Dropout: 0.1
  - Hidden continuous size: 32
  - Output: 7 quantiles (p10, p20, p30, p50, p70, p80, p90)
  - Loss: QuantileLoss for uncertainty estimation

- **Training**:
  - PyTorch Lightning Trainer
  - Max epochs: 50
  - Early stopping (patience=10)
  - Learning rate reduction on plateau
  - Gradient clipping (0.1)
  - GPU/CPU auto-detection
  - TensorBoard logging

- **Evaluation**:
  - Validation and test metrics (MAE, RMSE, MAPE)
  - Prediction visualizations with uncertainty bands
  - 4 random test samples shown with quantile predictions

- **Future Inference**:
  - 7-day (168-hour) future forecasting
  - Iterative prediction in 24-hour windows
  - Historical context visualization (30 days + 7 days future)
  - Hourly pattern comparison (historical vs forecast)

- **Interpretability**:
  - Variable importance (encoder, decoder, static)
  - Attention weight visualization (heatmaps)
  - Temporal attention analysis (recent vs distant past)
  - Top 5 most important features identified

## Key Implementation Details

### Feature Engineering
- **Cyclical encoding**: sin/cos for periodic features (hour, month, day of week)
- **Lag features**: Essential for autoregressive forecasting
- **Rolling statistics**: Capture recent trends
- **Proper indexing**: All features use time_idx for TFT compatibility

### Train/Val/Test Splits
- **Shallow models**: 80/20 train/test with temporal ordering
- **TFT**: 70/15/15 train/val/test with proper TimeSeriesDataSet.from_dataset()

### Variance Analysis
All pattern visualizations (hourly, daily, seasonal) include:
- Mean values (primary metric)
- ±1 standard deviation (shaded regions or error bars)
- Variance statistics printed in tables

### Model Checkpointing
- TFT automatically saves best model based on validation loss
- Checkpoint path available in trainer.checkpoint_callback.best_model_path
- Load with: TemporalFusionTransformer.load_from_checkpoint(path)

## Important Notes for Future Development

### When Adding New Features
1. Add to TimeSeriesDataSet configuration under appropriate category:
   - `time_varying_known_categoricals`: Future known categorical (hour, month)
   - `time_varying_known_reals`: Future known continuous (cyclical encodings)
   - `time_varying_unknown_reals`: Only known until present (energy, lags)
   - `static_categoricals`: Constant per series (series_id)

### When Training TFT
1. Ensure data has no gaps in time_idx (use allow_missing_timesteps=False)
2. Use GroupNormalizer for target normalization
3. Monitor validation loss, not training loss
4. Training takes 10-30 minutes on GPU, 1-2 hours on CPU
5. Check attention weights to verify model is learning meaningful patterns

### When Making Future Predictions
1. Must provide all time_varying_known features for future period
2. Unknown features can be left as NaN (model handles internally)
3. Use iterative prediction for horizons > max_prediction_length
4. Uncertainty bands come from quantile predictions

### Common Issues
- **Index misalignment**: Always use .iloc[] for temporal splits
- **Missing values**: TFT handles NaN in unknown features, but not in known features
- **Memory**: Reduce batch_size if OOM errors occur
- **Convergence**: If loss doesn't decrease, check learning rate and normalization

## Performance Benchmarks

Based on the current implementation:
- **Shallow models**: Train in seconds to minutes
- **TFT training**: ~20 epochs typical, ~30 minutes on GPU
- **Inference**: Real-time for single predictions, batch for efficiency
- **Metrics**: RMSE typically 500-2000 MW (depends on region and horizon)
