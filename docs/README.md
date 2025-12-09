# Documentation

Complete documentation for the Energy Consumption Forecasting project.

## Model Guides

### Deep Learning Models
- **[LSTM/GRU Guide](LSTM_GUIDE.md)** - Recommended starting point
  - Simple, fast, and effective recurrent neural networks
  - Comprehensive hyperparameter tuning guide
  - GPU acceleration support
  - Expected training time: 5-15 minutes

- **[TFT Guide](TFT_GUIDE.md)** - Advanced transformer model
  - State-of-the-art attention-based architecture
  - Interpretable predictions with uncertainty quantification
  - Longer training time, highest accuracy

### Traditional ML Models
Traditional ML models (XGBoost, LightGBM, CatBoost, Random Forest) are implemented in `src/modeling.py` and demonstrated in the exploration notebook. These models provide robust baselines and valuable insights through feature importance.

#### Shallow Model Predictions
![Shallow Model Predictions](../figures/shallow_predictions.png)
*Example predictions from traditional machine learning models, showcasing their forecasting capabilities.*

#### Shallow Model Scatter Plot
![Shallow Model Scatter Plot](../figures/shallow_scatter.png)
*A scatter plot comparing actual vs. predicted values for shallow models, indicating prediction accuracy.*

#### Feature Importance
![Feature Importance](../figures/feature_importance.png)
*Feature importance derived from tree-based models (e.g., XGBoost), highlighting key drivers of energy consumption.*


## Exploratory Data Analysis Visualizations
Understanding the underlying patterns in energy consumption is crucial for accurate forecasting. These visualizations highlight key temporal trends.

### Daily Consumption Patterns
![Daily Consumption Patterns](../figures/daily_patterns.png)
*Average hourly energy consumption patterns across different days of the week, revealing typical usage profiles.*

### Hourly Consumption Patterns
![Hourly Consumption Patterns](../figures/hourly_patterns.png)
*Detailed hourly consumption trends, often showing peaks during morning and evening hours.*

### Seasonal Consumption Patterns
![Seasonal Consumption Patterns](../figures/seasonal_patterns.png)
*Energy consumption trends observed across different seasons or months of the year, influenced by weather and daylight variations.*

## Project Documentation

- **[Getting Started](GETTING_STARTED.md)** - Quick start guide
- **[Data Guide](DATA_GUIDE.md)** - Dataset and features
- **[Model Comparison](MODEL_COMPARISON.md)** - Performance comparison
- **[Visualisation Guide](VISUALIZATION_GUIDE.md)** - Interpreting figures

## Repository Structure

```
EnergyConsumption/
├── docs/               # Documentation
├── notebooks/          # Jupyter notebooks
├── scripts/            # Training and visualisation scripts
├── src/                # Core library
├── figures/            # Generated visualisations
└── checkpoints/        # Model checkpoints
```

## Quick Navigation

**Beginners:**
1. [Getting Started](GETTING_STARTED.md)
2. [LSTM/GRU Guide](LSTM_GUIDE.md)
3. [Visualisation Guide](VISUALIZATION_GUIDE.md)

**Advanced Users:**
1. [Model Comparison](MODEL_COMPARISON.md)
2. [TFT Guide](TFT_GUIDE.md)
3. [Data Guide](DATA_GUIDE.md)

**Researchers:**
1. [Analysis Report](../Analysis.md)
2. Exploration notebook
3. [Model Comparison](MODEL_COMPARISON.md)
