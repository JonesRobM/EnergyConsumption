# Documentation

Complete documentation for the Energy Consumption Forecasting project.

## Model Guides

### Deep Learning Models
- **[LSTM/GRU Guide](LSTM_GUIDE.md)** - Recommended starting point for time-series forecasting
  - Simple, fast, and effective recurrent neural networks
  - Comprehensive hyperparameter tuning guide
  - GPU acceleration support
  - Expected training time: 5-15 minutes

- **[TFT Guide](TFT_GUIDE.md)** - Advanced temporal fusion transformer
  - State-of-the-art attention-based model
  - Interpretable predictions with uncertainty quantification
  - Longer training time but highest accuracy

### Traditional ML Models
Traditional machine learning models (XGBoost, LightGBM, CatBoost, Random Forest) are implemented in `src/modeling.py` and demonstrated in the exploration notebook.

## Project Documentation

- **[Getting Started](GETTING_STARTED.md)** - Quick start guide for new users
- **[Data Guide](DATA_GUIDE.md)** - Understanding the dataset and features
- **[Model Comparison](MODEL_COMPARISON.md)** - Performance comparison and selection guide
- **[Visualization Guide](VISUALIZATION_GUIDE.md)** - Understanding the generated figures

## Repository Structure

```
EnergyConsumption/
├── docs/               # Documentation (you are here)
├── notebooks/          # Jupyter notebooks for analysis
├── scripts/            # Training and figure generation scripts
├── src/                # Core library code
├── figures/            # Generated visualizations
└── checkpoints/        # Trained model checkpoints
```

## Quick Navigation

**For Beginners:**
1. Start with [Getting Started](GETTING_STARTED.md)
2. Follow the [LSTM/GRU Guide](LSTM_GUIDE.md)
3. Understand your results with [Visualization Guide](VISUALIZATION_GUIDE.md)

**For Advanced Users:**
1. Review [Model Comparison](MODEL_COMPARISON.md)
2. Explore [TFT Guide](TFT_GUIDE.md) for best accuracy
3. Check [Data Guide](DATA_GUIDE.md) for feature engineering

**For Researchers:**
1. Review the detailed [Analysis](../Analysis.md)
2. Examine the exploration notebook
3. Compare model architectures in [Model Comparison](MODEL_COMPARISON.md)
