# TFT Training and Testing Guide

Complete guide for training, testing, and using the Temporal Fusion Transformer (TFT) model for energy consumption forecasting.

## Quick Start

### Option 1: Using Jupyter Notebook (Recommended for exploration)
```bash
jupyter notebook exploration.ipynb
```
Navigate to the "Temporal Fusion Transformer" section and run cells sequentially.

### Option 2: Using Standalone Scripts (Recommended for production)

#### 1. Train a new model
```bash
python train_tft.py --mode train --epochs 50 --batch_size 64
```

#### 2. Test an existing model
```bash
python train_tft.py --mode test --checkpoint checkpoints/tft-epoch=20-val_loss=0.45.ckpt
```

#### 3. Train and test in one go
```bash
python train_tft.py --mode train_test --epochs 30
```

#### 4. Make predictions and visualize
```bash
python predict_tft.py --checkpoint checkpoints/best_model.ckpt --n_samples 10
```

## Script Reference

### train_tft.py

**Purpose**: Train and/or test TFT models with full control over hyperparameters.

**Arguments**:
- `--mode`: Operation mode
  - `train`: Train a new model
  - `test`: Test existing model (requires --checkpoint)
  - `train_test`: Train and immediately test (default)

- `--data`: Path to CSV file (default: `composite_energy_data.csv`)
- `--region`: Energy region to forecast (default: `PJME_MW`)
- `--epochs`: Maximum training epochs (default: 50)
- `--batch_size`: Training batch size (default: 64)
- `--hidden_size`: Model hidden dimension (default: 64)
- `--attention_heads`: Number of attention heads (default: 4)
- `--checkpoint`: Model checkpoint path (required for `test` mode)

**Examples**:
```bash
# Quick training with defaults
python train_tft.py

# Custom hyperparameters
python train_tft.py --hidden_size 128 --attention_heads 8 --epochs 100

# Different region
python train_tft.py --region AEP_MW --epochs 40

# Test specific checkpoint
python train_tft.py --mode test --checkpoint checkpoints/tft-epoch=25-val_loss=0.38.ckpt
```

**Output**:
- Checkpoints saved in `checkpoints/`
- TensorBoard logs in `lightning_logs/`
- Prints training progress and final test metrics

### predict_tft.py

**Purpose**: Load trained model and make predictions with visualizations.

**Arguments**:
- `--checkpoint`: Path to model checkpoint (required)
- `--n_samples`: Number of sample plots to generate (default: 5)
- `--future`: Make true future predictions beyond available data (experimental)

**Examples**:
```bash
# Generate 5 sample prediction plots
python predict_tft.py --checkpoint checkpoints/tft-epoch=20-val_loss=0.45.ckpt

# Generate 20 samples
python predict_tft.py --checkpoint checkpoints/best_model.ckpt --n_samples 20
```

**Output**:
- PNG plots: `prediction_sample_1.png`, `prediction_sample_2.png`, etc.
- Test metrics: MAE, RMSE, MAPE

## Understanding the Pipeline

### Data Flow

```
Raw Data (CSV)
    ↓
Load & Clean (composite_energy_data.csv)
    ↓
Feature Engineering
    - Temporal: hour, month, dayofweek
    - Cyclical: sin/cos encodings
    - Lags: 24hr, 168hr
    - Rolling: 24hr mean, 168hr mean
    ↓
TimeSeriesDataSet Creation
    - Train: 70%
    - Validation: 15%
    - Test: 15%
    - Encoder length: 168 hours (7 days)
    - Prediction length: 24 hours (1 day)
    ↓
TFT Model Training
    - PyTorch Lightning Trainer
    - Early stopping (patience=10)
    - Learning rate reduction
    - Gradient clipping
    ↓
Evaluation & Prediction
    - Quantile predictions (uncertainty)
    - Variable importance
    - Attention weights
```

### Model Architecture

**Temporal Fusion Transformer** combines:
1. **Gating mechanisms**: Select relevant features
2. **Variable selection**: Identify important inputs
3. **Temporal processing**: LSTM for sequence modeling
4. **Multi-head attention**: Capture long-range dependencies
5. **Quantile forecasting**: Uncertainty estimation

**Default Configuration**:
- Hidden size: 64
- Attention heads: 4
- Dropout: 0.1
- Quantiles: [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
- Loss: QuantileLoss

## Training Tips

### 1. Start Small
```bash
# Fast training for debugging (5-10 min)
python train_tft.py --epochs 10 --batch_size 128
```

### 2. Monitor Training
```bash
# Launch TensorBoard
tensorboard --logdir lightning_logs

# Open browser to http://localhost:6006
```

### 3. GPU vs CPU
- **GPU**: ~20-30 minutes for 50 epochs (recommended)
- **CPU**: ~1-2 hours for 50 epochs (works but slower)
- Script auto-detects GPU availability

### 4. Hyperparameter Tuning

**If loss doesn't decrease**:
- Reduce learning rate: Add `learning_rate=0.0001` in `create_model()`
- Increase batch size: `--batch_size 128`
- Check for data leakage or NaN values

**If underfitting (high training loss)**:
- Increase model capacity: `--hidden_size 128 --attention_heads 8`
- Train longer: `--epochs 100`
- Add more features (in `load_and_prepare_data()`)

**If overfitting (val_loss > train_loss)**:
- Increase dropout: Modify `dropout=0.2` in `create_model()`
- Reduce model size: `--hidden_size 32`
- Early stopping will help automatically

### 5. Memory Issues

If you get OOM (Out of Memory) errors:
```bash
# Reduce batch size
python train_tft.py --batch_size 32

# Or modify encoder/prediction length in create_datasets()
# encoder_length=120 (instead of 168)
# prediction_length=12 (instead of 24)
```

## Interpreting Results

### Training Metrics
- **train_loss**: Should decrease steadily
- **val_loss**: Should track train_loss (with ~10% gap acceptable)
- **Early stopping**: Triggered if val_loss doesn't improve for 10 epochs

### Test Metrics
- **MAE** (Mean Absolute Error): Average prediction error in MW
  - Good: < 1000 MW
  - Excellent: < 500 MW

- **RMSE** (Root Mean Squared Error): Penalizes large errors
  - Good: < 1500 MW
  - Excellent: < 800 MW

- **MAPE** (Mean Absolute Percentage Error): Relative error
  - Good: < 5%
  - Excellent: < 2%

### Prediction Plots
Each plot shows:
- **Blue line**: Historical context (168 hours)
- **Green line**: Actual future values
- **Red dashed**: Predicted median (50th percentile)
- **Red shaded**: 10-90% prediction interval

**Good predictions**:
- Red dashed closely follows green line
- Shaded region contains green line ~80% of the time
- Captures daily patterns and trends

## Finding Best Checkpoint

Checkpoints are automatically saved in `checkpoints/` with format:
```
tft-epoch=XX-val_loss=Y.YY.ckpt
```

**Find best model**:
```bash
# Windows
dir checkpoints /O:N

# Look for lowest val_loss value
```

The model with **lowest val_loss** is typically best for testing.

## Common Issues & Solutions

### Issue: "Data file not found"
**Solution**: Run `exploration.ipynb` notebook first to generate `composite_energy_data.csv`

### Issue: "CUDA out of memory"
**Solution**:
```bash
python train_tft.py --batch_size 32
# or force CPU
python train_tft.py --batch_size 64  # Will auto-detect CPU if no GPU
```

### Issue: "TypeError: model must be a LightningModule"
**Solution**: Package version mismatch. Run:
```bash
.venv\Scripts\activate
pip install "lightning>=2.1.0" "pytorch-forecasting>=1.0.0" --force-reinstall
```

### Issue: Training is very slow
**Solutions**:
- Use smaller model: `--hidden_size 32 --attention_heads 2`
- Smaller batch size paradoxically faster on CPU: `--batch_size 32`
- Reduce data: Modify `train_ratio=0.5` in `create_datasets()`

### Issue: Predictions are flat/boring
**Possible causes**:
- Model collapsed to predicting mean
- Check if variance in data is too small
- Increase model capacity or train longer
- Verify features have meaningful signal

## Advanced Usage

### Custom Region Analysis
```python
# Edit train_tft.py, change default region
python train_tft.py --region COMED_MW --epochs 50
```

### Batch Hyperparameter Search
Create a bash script:
```bash
# hyperparameter_search.sh
for hidden in 32 64 128; do
    for heads in 2 4 8; do
        python train_tft.py --hidden_size $hidden --attention_heads $heads --epochs 30
    done
done
```

### Export Predictions to CSV
Modify `predict_tft.py` to save:
```python
# Add after line where predictions are made
df_preds = pd.DataFrame({
    'actual': actuals_flat,
    'predicted': preds_flat
})
df_preds.to_csv('predictions.csv', index=False)
```

## Next Steps

1. **Baseline**: Run default training to establish performance
   ```bash
   python train_tft.py --mode train_test
   ```

2. **Visualize**: Generate prediction samples
   ```bash
   python predict_tft.py --checkpoint checkpoints/[BEST_MODEL].ckpt
   ```

3. **Optimize**: Try different hyperparameters based on results

4. **Productionize**: See `GEMINI.md` for deployment ideas (FastAPI, MLflow, etc.)

## Reference

- **PyTorch Forecasting Docs**: https://pytorch-forecasting.readthedocs.io/
- **TFT Paper**: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2021)
- **Lightning Docs**: https://lightning.ai/docs/pytorch/stable/

---

**Questions?** Check `exploration.ipynb` for detailed explanations of each step.
