# Energy Consumption Analysis

A comprehensive data science project for analyzing and predicting energy consumption patterns using machine learning and deep learning techniques. This project uses the **Kaggle Hourly Energy Consumption Dataset** to build interpretable predictive models using traditional ML algorithms and state-of-the-art Temporal Fusion Transformers (TFT).

## Overview

This project provides a complete pipeline for energy consumption forecasting:
- **Exploratory Data Analysis (EDA)** with statistical variance analysis
- **Correlation Studies** (month-to-month, year-over-year)
- **Feature Engineering** with temporal and cyclical features
- **Shallow Learning Models** (Linear Regression, Ridge, Lasso, Random Forest)
- **Gradient Boosting** (XGBoost, LightGBM, CatBoost)
- **Deep Learning** with Temporal Fusion Transformers for interpretable multi-horizon forecasting
- **Model Evaluation** with comprehensive metrics (MAE, RMSE, MAPE, R²)
- **Future Inference** for 7-day forecasting beyond historical data

## Key Features

### 📊 **Comprehensive Analysis**
- **Statistical Testing**: Distribution analysis, normality tests (Anderson-Darling), skewness/kurtosis
- **Temporal Patterns**: Hourly, daily, weekly, and seasonal consumption patterns with variance
- **Correlation Analysis**: Cross-regional correlation, month-by-month patterns, year-over-year consistency
- **Time Series Decomposition**: Trend, seasonal, and residual component analysis

### 🤖 **Multiple Modeling Approaches**
- **Traditional ML**: Linear models with regularization (Ridge, Lasso)
- **Ensemble Methods**: Random Forest with feature importance
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost with hyperparameter tuning
- **Deep Learning**: Temporal Fusion Transformer with:
  - Multi-head attention mechanisms
  - Quantile regression for uncertainty estimation
  - Variable selection networks for interpretability
  - 168-hour encoder (7-day lookback), 24-hour decoder (1-day forecast)

### 🎯 **Production-Ready Pipeline**
- Train/Validation/Test splits (70/15/15)
- Early stopping and learning rate scheduling
- Model checkpointing and versioning
- GPU/CPU automatic detection
- Comprehensive evaluation metrics
- Attention visualization for model interpretability

### 📈 **Visualization Suite**
- Interactive plots with Matplotlib, Seaborn, and Plotly
- Prediction interval visualization (50%, 80% confidence)
- Attention heatmaps showing temporal dependencies
- Variable importance rankings
- Historical vs future forecast comparison

## Installation

### Prerequisites
- Python 3.9 or higher
- Virtual environment (`.venv` included)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd EnergyConsumption
```

2. Activate the virtual environment:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Kaggle API (for dataset downloads):
   - Go to https://www.kaggle.com/account
   - Create a new API token
   - Place `kaggle.json` in `~/.kaggle/` (or `C:\Users\<username>\.kaggle\` on Windows)

## Quick Start

You have **two options** for running the analysis:

### Option 1: Jupyter Notebook (Interactive Exploration)
```bash
jupyter notebook exploration.ipynb
```

The `exploration.ipynb` notebook contains the complete analysis pipeline:

**Data Preparation**
- Dataset download and loading from Kaggle
- Data preprocessing and feature engineering
- Temporal feature extraction (cyclical encodings)

**Exploratory Analysis**
- Month-to-month and year-over-year correlations
- Statistical analysis with variance
- Hourly, daily, and seasonal pattern visualization

**Shallow Learning**
- Traditional ML models (Linear, Ridge, Lasso, Random Forest)
- Gradient boosting (XGBoost, LightGBM, CatBoost)
- Feature importance analysis
- Model comparison

**Deep Learning (TFT)**
- Temporal Fusion Transformer setup and training
- Multi-horizon forecasting (24-hour predictions)
- Model evaluation and validation
- Future inference (7-day forecasting)
- Interpretability analysis (attention weights, variable importance)

### Option 2: Standalone Scripts (Automated Training/Testing)

**NEW**: Train and test TFT models with standalone Python scripts!

#### 1. Train a TFT model
```bash
python train_tft.py --mode train --epochs 50
```

#### 2. Test an existing model
```bash
python train_tft.py --mode test --checkpoint checkpoints/tft-epoch=20-val_loss=0.45.ckpt
```

#### 3. Make predictions with visualizations
```bash
python predict_tft.py --checkpoint checkpoints/tft-epoch=20-val_loss=0.45.ckpt --n_samples 10
```

#### 4. Monitor training with TensorBoard
```bash
tensorboard --logdir lightning_logs
# Open http://localhost:6006
```

**See [TFT_GUIDE.md](TFT_GUIDE.md) for complete usage instructions and hyperparameter tuning.**

### Alternative: Download Dataset Separately
```bash
kaggle datasets download -d robikscube/hourly-energy-consumption
```

## Technology Stack

### Core Libraries
- **NumPy** (>=1.24.0) - Numerical computing and array operations
- **Pandas** (>=2.0.0) - Data manipulation and time series analysis
- **SciPy** (>=1.10.0) - Scientific computing and statistical functions

### Visualization
- **Matplotlib** (>=3.7.0) - Static plotting and visualizations
- **Seaborn** (>=0.12.0) - Statistical data visualization
- **Plotly** (>=5.14.0) - Interactive web-based visualizations

### Machine Learning (Shallow Learning)
- **scikit-learn** (>=1.3.0) - Traditional ML algorithms (regression, classification)
- **XGBoost** (>=2.0.0) - Gradient boosting with regularization
- **LightGBM** (>=4.0.0) - Fast gradient boosting framework
- **CatBoost** (>=1.2.0) - Gradient boosting with categorical feature support
- **statsmodels** (>=0.14.0) - Statistical models and time series decomposition

### Deep Learning & Time Series
- **PyTorch** (>=2.1.0) - Dynamic deep learning framework
- **PyTorch Lightning** (>=2.1.0) - High-level PyTorch training framework
- **PyTorch Forecasting** (>=1.0.0) - Time series forecasting with deep learning
- **TensorFlow** (>=2.15.0) - Neural network framework (included for flexibility)

### Development Tools
- **Jupyter** (>=1.0.0) - Interactive notebook environment
- **Kaggle API** (>=1.5.0) - Dataset download and competition integration
- **tqdm** (>=4.65.0) - Progress bars for long-running operations
- **python-dotenv** (>=1.0.0) - Environment variable management

## Project Structure

```
EnergyConsumption/
├── exploration.ipynb           # Main analysis notebook (complete pipeline)
├── train_tft.py                # Standalone TFT training/testing script
├── predict_tft.py              # TFT prediction and visualization script
├── composite_energy_data.csv   # Processed dataset (auto-generated)
├── requirements.txt            # Project dependencies
├── pyproject.toml              # Project configuration and metadata
├── TFT_GUIDE.md                # Complete TFT training guide
├── CLAUDE.md                   # Development guide for Claude Code
├── GEMINI.md                   # Project status guide for Gemini
├── README.md                   # This file
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore patterns
├── .venv/                      # Virtual environment (local)
├── checkpoints/                # Trained model checkpoints (auto-generated)
└── lightning_logs/             # TFT training logs (auto-generated)
```

### Main Notebook Sections

`exploration.ipynb` is organized into the following sections:

1. **Data Loading & Preprocessing**
   - Kaggle dataset download
   - Data aggregation and cleaning
   - Feature engineering

2. **Exploratory Data Analysis**
   - Cross-regional correlation analysis
   - Month-to-month correlation patterns
   - Year-over-year consistency analysis
   - Statistical distribution analysis
   - Temporal pattern visualization (hourly, daily, seasonal) with variance

3. **Shallow Learning Models**
   - Data preparation with lag and rolling features
   - Linear models (Linear Regression, Ridge, Lasso)
   - Random Forest
   - Gradient boosting (XGBoost, LightGBM, CatBoost)
   - Feature importance analysis
   - Model comparison and evaluation

4. **Temporal Fusion Transformer**
   - TFT data preparation with TimeSeriesDataSet
   - Model architecture configuration
   - Training with PyTorch Lightning
   - Validation and testing
   - Future inference (7-day forecasting)
   - Interpretability analysis (attention weights, variable importance)

## Results & Performance

The project implements multiple modeling approaches with varying levels of complexity:

### Shallow Learning Models
- **Linear Models**: Fast baseline with interpretable coefficients
- **Random Forest**: Good performance with automatic feature interactions
- **Gradient Boosting**: Best shallow learning performance
  - XGBoost, LightGBM, and CatBoost all achieve competitive results
  - Feature importance reveals lag features as most predictive

### Temporal Fusion Transformer
- **Architecture**: 64 hidden units, 4 attention heads
- **Performance Metrics**: MAE, RMSE, MAPE on validation and test sets
- **Uncertainty Quantification**: 7 quantiles (p10, p20, ..., p90)
- **Interpretability**:
  - Variable importance rankings for encoder/decoder features
  - Attention weights showing which historical timesteps matter most
  - Typically focuses more on recent 24-48 hours for short-term forecasting
- **Future Forecasting**: Generates 168-hour (7-day) predictions with uncertainty bands

### Key Insights
- **Hourly patterns**: Clear peak consumption during business hours (9am-5pm)
- **Weekly patterns**: Lower consumption on weekends vs weekdays
- **Seasonal patterns**: Higher consumption in summer/winter (cooling/heating)
- **Variance analysis**: Peak hours show higher variance (less predictable)
- **Lag features**: Previous 24h and 168h (1 week) are most predictive

## Usage

### Training TFT Models

#### Quick Start
```bash
# Train with defaults (50 epochs, ~30-50 min on CPU)
python train_tft.py --mode train_test

# Quick test run (5 epochs, ~5-10 min)
python train_tft.py --epochs 5
```

#### Custom Hyperparameters
```bash
# Larger model
python train_tft.py --hidden_size 128 --attention_heads 8 --epochs 100

# Different region
python train_tft.py --region AEP_MW --epochs 50

# Custom batch size (reduce if OOM errors)
python train_tft.py --batch_size 32
```

#### Making Predictions
```bash
# Generate prediction plots
python predict_tft.py --checkpoint checkpoints/tft-epoch=20-val_loss=0.45.ckpt --n_samples 10

# This creates prediction_sample_1.png, prediction_sample_2.png, etc.
```

### Expected Performance (PJME Region)
- **Training Time**: 20-30 min (50 epochs, CPU) | 10-15 min (GPU)
- **RMSE**: 500-1500 MW on 24-hour forecast horizon
- **MAE**: 300-1000 MW
- **MAPE**: 2-5%

### Documentation
- **[TFT_GUIDE.md](TFT_GUIDE.md)**: Complete TFT training guide with troubleshooting
- **[CLAUDE.md](CLAUDE.md)**: Development instructions and technical details
- **[GEMINI.md](GEMINI.md)**: Project status, challenges, and future directions

## Contributing

This project uses:
- **Black** for code formatting (line length: 100)
- **Ruff** for fast linting
- **MyPy** for type checking (optional)

Install development dependencies:
```bash
pip install -e ".[dev]"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: [Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) from Kaggle
- **TFT Implementation**: [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/)
- **Framework**: PyTorch Lightning for efficient training