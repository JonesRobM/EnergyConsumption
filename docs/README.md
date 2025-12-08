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
Traditional ML models (XGBoost, LightGBM, CatBoost, Random Forest) are implemented in `src/modeling.py` and demonstrated in the exploration notebook.

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
