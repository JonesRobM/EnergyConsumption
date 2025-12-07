# Model Comparison Guide

Complete comparison of all available models for energy consumption forecasting.

## Performance Summary

| Model | RMSE (MW) | Training Time | Complexity | GPU Required | Best For |
|-------|-----------|---------------|------------|--------------|----------|
| **Linear Regression** | ~1500 | Seconds | Very Low | No | Baseline comparison |
| **Ridge/Lasso** | ~1500 | Seconds | Low | No | Feature selection |
| **Random Forest** | ~1100 | 1-2 min | Medium | No | Quick results |
| **XGBoost** | ~900 | 2-5 min | Medium | No | Fast, accurate results |
| **LightGBM** | ~900 | 2-5 min | Medium | No | Large datasets |
| **CatBoost** | ~950 | 3-7 min | Medium | No | Robust performance |
| **LSTM** | ~800-1200 | 5-15 min | High | Recommended | Balance of speed/accuracy |
| **GRU** | ~800-1000 | 5-15 min | High | Recommended | Often best overall |
| **TFT** | ~700-1000 | 30-60 min | Very High | Required | Best accuracy |

**Note:** RMSE values are approximate and vary by region, configuration, and hyperparameters.

## Decision Tree

### Choose Your Model

```
Start Here
    │
    ├─ Need quick results (< 5 min)?
    │   └─ Yes → XGBoost or LightGBM
    │   └─ No → Continue
    │
    ├─ Have GPU available?
    │   └─ No → XGBoost/LightGBM
    │   └─ Yes → Continue
    │
    ├─ Want best accuracy?
    │   └─ Yes → TFT (if time allows)
    │   └─ No → Continue
    │
    └─ Want good balance?
        └─ **GRU (Recommended)**
```

## Detailed Model Analysis

### Traditional Machine Learning

#### Linear Models (Ridge, Lasso, Linear Regression)
**Pros:**
- Extremely fast training (seconds)
- Easy to interpret
- No GPU required
- Good baseline

**Cons:**
- Poor accuracy on non-linear patterns
- Can't capture complex temporal dependencies
- Limited forecasting capability

**When to use:**
- Quick baseline comparison
- Feature importance analysis
- Simple linear trends

#### Random Forest
**Pros:**
- Fast training (1-2 minutes)
- Handles non-linear patterns
- Built-in feature importance
- No scaling required

**Cons:**
- Moderate accuracy
- Can overfit with wrong parameters
- Slower than gradient boosting

**When to use:**
- Quick non-linear model
- Feature importance needed
- No hyperparameter tuning time

#### XGBoost / LightGBM / CatBoost
**Pros:**
- Excellent accuracy (RMSE ~900 MW)
- Fast training (2-5 minutes)
- No GPU required
- Robust to hyperparameters
- Feature importance included

**Cons:**
- Requires feature engineering
- Not designed for sequences
- Limited long-term dependencies

**When to use:**
- **Best choice if no GPU available**
- Need fast, accurate results
- Feature engineering is done
- Production deployment

**Recommendation:** Start with LightGBM for best speed/accuracy balance.

### Deep Learning Models

#### LSTM (Long Short-Term Memory)
**Pros:**
- Good accuracy (RMSE ~800-1200 MW)
- Designed for sequences
- Captures long-term dependencies
- Moderate training time (5-15 min with GPU)

**Cons:**
- Requires GPU for reasonable speed
- More complex than traditional ML
- Requires more data
- Can be harder to tune

**When to use:**
- Have GPU available
- Want neural network approach
- Long sequences (weeks/months)
- Good balance of speed and accuracy

**Configuration:** See [LSTM/GRU Guide](LSTM_GUIDE.md)

#### GRU (Gated Recurrent Unit)
**Pros:**
- Often better than LSTM (RMSE ~800-1000 MW)
- Faster training than LSTM
- Simpler architecture = less overfitting
- Better generalization
- Same sequence modeling power

**Cons:**
- Same as LSTM
- Requires GPU for speed

**When to use:**
- **Recommended over LSTM in most cases**
- Same use cases as LSTM
- Want faster training
- Prefer simpler model

**Configuration:** Use `--use_gru` flag with train_lstm.py

#### TFT (Temporal Fusion Transformer)
**Pros:**
- Best accuracy (RMSE ~700-1000 MW)
- Interpretable attention mechanisms
- Uncertainty quantification
- Handles multiple time scales
- Static and dynamic features

**Cons:**
- Longest training time (30-60 min)
- Requires GPU
- Most complex to tune
- Needs more data

**When to use:**
- Need best possible accuracy
- Have time for training
- Want interpretability
- Production deployment with quality requirements

**Configuration:** See [TFT Guide](TFT_GUIDE.md)

## Recommendations by Use Case

### Use Case 1: Research & Exploration
**Goal:** Understand data and establish baseline

**Recommended Workflow:**
1. Start with **XGBoost** (5 minutes)
2. Check feature importance
3. Try **GRU** for comparison (15 minutes)
4. Analyze differences

**Total time:** 20-30 minutes

### Use Case 2: Production Deployment (CPU only)
**Goal:** Best accuracy without GPU

**Recommended Model:** **LightGBM**
- RMSE: ~900 MW
- Training: 2-5 minutes
- Inference: Very fast
- No GPU needed

**Alternative:** CatBoost (more robust)

### Use Case 3: Production Deployment (GPU available)
**Goal:** Best accuracy with GPU

**Recommended Model:** **GRU**
- RMSE: ~800-1000 MW
- Training: 10-15 minutes
- Inference: Fast on GPU
- Good generalization

**Alternative:** TFT if you need interpretability

### Use Case 4: Academic/Competition
**Goal:** Maximum accuracy, no time constraints

**Recommended Workflow:**
1. Train **TFT** with full tuning (2-3 hours)
2. Train **GRU** ensemble (3 models, 1 hour)
3. Combine predictions

**Expected RMSE:** 650-800 MW

### Use Case 5: Quick Prototype
**Goal:** Something working in 5 minutes

**Recommended Model:** **XGBoost** or **LightGBM**
```bash
# Already implemented in src/modeling.py
# Run exploration.ipynb cells
```

## Feature Requirements

### Traditional ML (XGBoost, LightGBM, CatBoost)
**Required Features:**
- Temporal features (hour, day, month)
- Lag features (previous values)
- Rolling statistics (moving averages)
- Calendar features (holiday, weekend)

**Feature engineering done in:** `src/feature_engineering.py`

### Deep Learning (LSTM, GRU, TFT)
**Automatic Features:**
- Temporal encoding built-in
- Learns from raw sequences
- Minimal feature engineering needed

**Still beneficial:**
- External features (weather, holidays)
- Categorical encodings
- Domain-specific features

## Ensemble Approaches

### Option 1: Simple Average
Train multiple models and average predictions:
```python
# Train 3 different models
model1 = GRU (hidden=256)
model2 = GRU (hidden=512)
model3 = LightGBM

final_prediction = (pred1 + pred2 + pred3) / 3
```

**Expected improvement:** 5-10% RMSE reduction

### Option 2: Weighted Average
Weight predictions by validation performance:
```python
weights = [0.4, 0.4, 0.2]  # Based on val RMSE
final_prediction = w1*pred1 + w2*pred2 + w3*pred3
```

**Expected improvement:** 10-15% RMSE reduction

### Option 3: Stacking
Use model predictions as features for a meta-model:
```python
# Level 1: Base models
base_models = [GRU, LSTM, XGBoost]

# Level 2: Meta-model
meta_model = Ridge()  # Simple model to avoid overfitting
```

**Expected improvement:** 15-20% RMSE reduction (competition-level)

## Hyperparameter Impact

### Most Important to Least Important

**For LSTM/GRU:**
1. **hidden_size** (128 → 512): 30-40% improvement
2. **num_layers** (2 → 4): 15-20% improvement
3. **lookback** (72 → 336): 10-15% improvement
4. **learning_rate** (0.01 → 0.0005): 10-15% improvement
5. **dropout** (0.1 → 0.4): 5-10% improvement
6. **batch_size** (128 → 32): 5-10% improvement

**For XGBoost/LightGBM:**
1. **n_estimators** (100 → 500): 20-30% improvement
2. **max_depth** (5 → 10): 15-20% improvement
3. **learning_rate** (0.1 → 0.05): 10-15% improvement
4. **subsample** (1.0 → 0.8): 5-10% improvement

## Cost Analysis

### Training Cost (GPU: RTX 4050)

| Model | Single Run | Tuning (10 runs) | Power Cost* |
|-------|-----------|------------------|-------------|
| XGBoost | $0.01 | $0.10 | Negligible |
| GRU | $0.05 | $0.50 | ~$0.02 |
| TFT | $0.20 | $2.00 | ~$0.08 |

*Assuming $0.12/kWh, 36W TDP

### Inference Cost

| Model | Latency | Throughput | GPU Needed |
|-------|---------|------------|------------|
| XGBoost | <1ms | 10,000/sec | No |
| GRU | ~5ms | 200/sec | Optional |
| TFT | ~20ms | 50/sec | Recommended |

## Summary Recommendations

**Default Choice:** **GRU with 512 hidden units**
- Best overall accuracy/speed trade-off
- Works well out of the box
- Easy to tune further

**No GPU:** **LightGBM**
- Second-best accuracy
- Very fast
- Production-ready

**Maximum Accuracy:** **TFT**
- Best results
- Worth the training time
- Use for critical applications

**Quick Prototype:** **XGBoost**
- Fast to train
- Good enough for most cases
- Available in exploration notebook
