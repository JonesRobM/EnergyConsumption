# LSTM/GRU Model Guide

This guide provides comprehensive documentation for training and evaluating LSTM/GRU models for energy consumption forecasting using the `train_lstm.py` script.

## Overview

LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) are recurrent neural network architectures designed for sequence modelling. Unlike the more complex Temporal Fusion Transformer (TFT), these models offer:

- **Simplicity**: Easier to understand and debug
- **Faster training**: Typically trains faster than attention-based models
- **Lower memory requirements**: More efficient for large datasets
- **Strong baseline performance**: Often achieves competitive results with less complexity

## Model Architecture

The implementation includes:

- **Configurable depth**: 1-4 LSTM/GRU layers (default: 2)
- **Hidden size**: 64-512 units per layer (default: 128)
- **Dropout**: Regularization between layers (default: 0.2)
- **Sequence-to-vector**: Takes in a sequence and predicts future values
- **Adam optimiser**: With learning rate scheduling
- **Early stopping**: Prevents overfitting

### Features Used

The model uses the following engineered features:

**Temporal Features:**
- Hour of day (0-23)
- Day of week (0-6)
- Month (1-12)
- Day of month
- Quarter (1-4)

**Lag Features:**
- lag_1: Previous hour's value
- lag_24: Same hour yesterday
- lag_168: Same hour last week

**Rolling Statistics:**
- rolling_mean_24: 24-hour moving average
- rolling_std_24: 24-hour moving standard deviation

## Quick Start

### Basic Training

Train an LSTM model with default settings:

```bash
python scripts/train_lstm.py --mode train_test --epochs 50
```

This will:
1. Load and preprocess the data
2. Create train/validation/test splits (70/15/15)
3. Train the LSTM model for 50 epochs
4. Evaluate on the test set
5. Save the model checkpoint to `checkpoints/lstm_best_PJME_MW.pt`
6. Generate training history and prediction plots in `figures/`

### Training a GRU Model

GRU models are often faster and perform similarly to LSTM:

```bash
python scripts/train_lstm.py --mode train_test --use_gru --epochs 50
```

### Custom Configuration

Train with custom hyperparameters:

```bash
python scripts/train_lstm.py \
  --mode train_test \
  --epochs 100 \
  --batch_size 128 \
  --hidden_size 256 \
  --num_layers 3 \
  --dropout 0.3 \
  --learning_rate 0.0005 \
  --lookback 168 \
  --forecast_horizon 24
```

## Command-Line Arguments

### General Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `train_test` | Mode of operation: `train`, `test`, or `train_test` |
| `--data_path` | str | `composite_energy_data.csv` | Path to the dataset |
| `--region` | str | `PJME_MW` | Energy region to forecast |
| `--checkpoint_path` | str | None | Path to checkpoint for testing |

### Model Architecture

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_gru` | flag | False | Use GRU instead of LSTM |
| `--hidden_size` | int | 128 | Hidden size for LSTM/GRU layers |
| `--num_layers` | int | 2 | Number of recurrent layers |
| `--dropout` | float | 0.2 | Dropout rate between layers |

### Training Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 50 | Maximum number of training epochs |
| `--batch_size` | int | 64 | Batch size for training |
| `--learning_rate` | float | 0.001 | Initial learning rate |
| `--patience` | int | 10 | Early stopping patience (epochs) |

### Time Series Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lookback` | int | 168 | Lookback window size (hours) |
| `--forecast_horizon` | int | 24 | Forecast horizon (hours) |

## Usage Examples

### 1. Train Only Mode

Train a model without testing:

```bash
python scripts/train_lstm.py --mode train --epochs 50
```

### 2. Test Only Mode

Test a previously trained model:

```bash
python scripts/train_lstm.py \
  --mode test \
  --checkpoint_path checkpoints/lstm_best_PJME_MW.pt
```

### 3. Different Regions

Train models for different energy regions:

```bash
# AEP region
python scripts/train_lstm.py --region AEP_MW --epochs 50

# DAYTON region
python scripts/train_lstm.py --region DAYTON_MW --epochs 50
```

### 4. Hyperparameter Tuning

**Small, fast model (for quick experiments):**
```bash
python scripts/train_lstm.py \
  --hidden_size 64 \
  --num_layers 1 \
  --epochs 30
```

**Large, powerful model (for best performance):**
```bash
python scripts/train_lstm.py \
  --hidden_size 512 \
  --num_layers 4 \
  --dropout 0.3 \
  --epochs 100 \
  --batch_size 32
```

**Long-term forecasting:**
```bash
python scripts/train_lstm.py \
  --lookback 336 \
  --forecast_horizon 72 \
  --epochs 75
```

## Understanding the Output

### Training Output

```
================================================================================
Training LSTM Model
================================================================================

[1/5] Loading data from composite_energy_data.csv...
   Loaded 145,366 rows for PJME_MW

[2/5] Engineering features...
   Final dataset: 145,198 rows

[3/5] Creating datasets...
   Train samples: 101,591
   Val samples: 21,733
   Test samples: 21,733

[4/5] Creating LSTM model...
   Model created: 206,872 parameters

[5/5] Training model...
   Epoch   1/50 | Train Loss: 0.2341 | Val Loss: 0.2156
   Epoch   2/50 | Train Loss: 0.1987 | Val Loss: 0.1923
   ...

   Training complete!
   Best validation loss: 0.1542
   Model saved to: checkpoints/lstm_best_PJME_MW.pt
   Training history plot saved to: figures/lstm_training_history.png
```

### Test Output

```
================================================================================
Testing Model
================================================================================

   Test Results:
   MSE:  1234.56
   RMSE: 35.14
   MAE:  27.89
   MAPE: 2.34%

   Testing complete!
   Predictions plot saved to: figures/lstm_predictions.png
```

## Output Files

### Checkpoints

Saved in `checkpoints/`:
- `lstm_best_{region}.pt` or `gru_best_{region}.pt`
- Contains model weights, optimiser state, scalers, and training history

### Figures

Saved in `figures/`:

**Training History** (`lstm_training_history.png` or `gru_training_history.png`):
- Training and validation loss curves over epochs
- Helps diagnose overfitting or underfitting

**Predictions** (`lstm_predictions.png` or `gru_predictions.png`):
- Top panel: Time series of predictions vs actual (last 500 points)
- Bottom panel: Scatter plot of actual vs predicted with metrics

## Checkpoint Structure

The saved checkpoint contains:

```python
{
    'model_state_dict': {...},      # Model weights
    'optimiser_state_dict': {...},  # Optimiser state
    'epoch': 42,                    # Final epoch number
    'val_loss': 0.1542,            # Best validation loss
    'config': {...},               # All command-line arguments
    'scaler_X': StandardScaler(),  # Feature scaler
    'scaler_y': StandardScaler(),  # Target scaler
    'train_losses': [...],         # Training loss history
    'val_losses': [...]            # Validation loss history
}
```

## Performance Expectations

### Typical Metrics (PJME_MW region)

With default hyperparameters (50 epochs, 168-hour lookback):

- **RMSE**: ~800-1200 MW
- **MAE**: ~600-900 MW
- **MAPE**: ~2-4%
- **Training time**: 5-15 minutes (CPU), 1-3 minutes (GPU)

### Comparison with Other Models

| Model | Complexity | Training Time | Typical RMSE |
|-------|-----------|---------------|--------------|
| Linear Regression | Low | Seconds | ~1500 MW |
| XGBoost/LightGBM | Medium | Minutes | ~900 MW |
| **LSTM/GRU** | **Medium-High** | **5-15 min** | **~800-1200 MW** |
| TFT | High | 30-60 min | ~700-1000 MW |

## Tips and Best Practices

### 1. Start Small

Begin with a small model and short training:
```bash
python scripts/train_lstm.py --hidden_size 64 --num_layers 1 --epochs 10
```

### 2. Monitor for Overfitting

Watch the training/validation loss gap:
- **Good**: Train and val losses decrease together
- **Overfitting**: Train loss decreases, val loss increases
- **Solution**: Increase dropout, reduce model size, or stop early

### 3. GRU vs LSTM

- **Try GRU first**: Often performs similarly with faster training
- **Use LSTM if**: You have very long sequences or complex patterns

### 4. Batch Size Selection

- **Smaller batches (32-64)**: Better generalisation, slower training
- **Larger batches (128-256)**: Faster training, may need higher learning rate

### 5. Learning Rate

- **Too high**: Training unstable, loss oscillates
- **Too low**: Slow convergence
- **Default (0.001)**: Good starting point for most cases

### 6. Sequence Length

- **Lookback window**:
  - 168 hours (1 week): Captures weekly patterns
  - 336 hours (2 weeks): Better for irregular patterns
  - Trade-off: Longer = more context but slower training

- **Forecast horizon**:
  - 24 hours (1 day): Standard short-term forecasting
  - 72 hours (3 days): Medium-term forecasting
  - Longer horizons are generally harder to predict

## Troubleshooting

### Problem: Model not improving

**Symptoms**: Validation loss stays constant or decreases very slowly

**Solutions**:
1. Increase learning rate: `--learning_rate 0.005`
2. Increase model capacity: `--hidden_size 256 --num_layers 3`
3. Check data preprocessing and feature engineering

### Problem: Overfitting

**Symptoms**: Train loss much lower than validation loss

**Solutions**:
1. Increase dropout: `--dropout 0.3` or `--dropout 0.4`
2. Reduce model size: `--hidden_size 64 --num_layers 1`
3. Reduce training epochs or use early stopping (automatic)
4. Get more training data

### Problem: Training too slow

**Solutions**:
1. Increase batch size: `--batch_size 128`
2. Use GRU instead of LSTM: `--use_gru`
3. Reduce sequence length: `--lookback 72`
4. Use GPU if available (automatic detection)

### Problem: Out of memory

**Solutions**:
1. Reduce batch size: `--batch_size 32`
2. Reduce model size: `--hidden_size 64`
3. Reduce sequence length: `--lookback 72`

## Advanced Usage

### Ensemble Predictions

Train multiple models and average predictions:

```bash
# Train 3 models with different configurations
python scripts/train_lstm.py --hidden_size 128 --num_layers 2 --epochs 50
python scripts/train_lstm.py --hidden_size 256 --num_layers 2 --epochs 50
python scripts/train_lstm.py --use_gru --hidden_size 192 --num_layers 3 --epochs 50

# Load and average predictions in Python
```

### Cross-Region Analysis

Train on one region, test on another:

```python
# Requires custom script modification
```

## Generating Figures

The `generate_figures.py` script can automatically generate visualisations from trained models:

```bash
python scripts/generate_figures.py
```

This will create all analysis figures plus LSTM/GRU training and prediction plots if checkpoints exist.

## Further Reading

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [GRU vs LSTM](https://arxiv.org/abs/1412.3555)
- [Time Series Forecasting with Deep Learning](https://arxiv.org/abs/1704.04110)

## Support

For issues or questions:
1. Check this guide for common solutions
2. Review the TFT_GUIDE.md for general time-series tips
3. Examine the generated figures for diagnostic information
4. Open an issue on the project repository
