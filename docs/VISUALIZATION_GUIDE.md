# Visualisation Guide

Complete guide to understanding and interpreting all generated figures and visualisations.

## Overview

This project automatically generates 13+ figures for data exploration, model training, and performance evaluation. All figures are saved in the `figures/` directory.

## Figure Categories

### 📊 Data Exploration (7 figures)
1. Cross-regional correlation
2. Monthly cross-regional correlation
3. Year-over-year correlation
4. Hourly consumption patterns
5. Daily consumption patterns
6. Seasonal consumption patterns
7. Time series decomposition

### 🤖 Model Performance (4 figures)
8. LSTM/GRU training history
9. LSTM/GRU predictions
10. Shallow model predictions
11. Shallow model scatter plots

### 🔍 Feature Analysis (2 figures)
12. Feature importance (XGBoost/LightGBM/CatBoost)
13. Regional consumption overview (optional)

## Data Exploration Figures

### 1. Cross-Regional Correlation (`cross_regional_correlation.png`)

**What it shows:**
- Correlation matrix between all regions
- Overall relationship across entire dataset

**How to read:**
- **1.0 (dark red)**: Perfect positive correlation
- **0.0 (white)**: No correlation
- **-1.0 (dark blue)**: Perfect negative correlation

**What to look for:**
- High correlations (>0.8): Regions with similar patterns
- Low correlations (<0.5): Regions with independent behaviour
- Block patterns: Regional groups with similar characteristics

**Insights:**
```
High correlation (>0.9):
- PJM regions tend to correlate strongly
- Nearby geographic regions correlate

Low correlation (<0.7):
- Different climate zones
- Different economic profiles
```

**Action items:**
- ✅ High correlation = can train single model for multiple regions
- ✅ Low correlation = need region-specific models

---

### 2. Monthly Cross-Regional Correlation (`monthly_cross_regional_correlation.png`)

**What it shows:**
- 4 subplots for January, April, July, October
- How regional correlations change by season

**How to read:**
- Compare same region pairs across months
- Look for seasonal variation in correlation strength

**What to look for:**
- **Summer (July)**: Often higher correlation (AC usage)
- **Winter (January)**: Correlation depends on heating type
- **Spring/Fall**: Moderate, transitional patterns

**Insights:**
```
Summer correlations typically higher because:
- Cooling dominated by electricity across regions
- Weather patterns more uniform
- Less variation in heating fuel mix
```

**Action items:**
- Consider seasonal models if correlation varies significantly
- Use month as feature in model

---

### 3. Year-over-Year Correlation (`yoy_correlation.png`)

**What it shows:**
- 4 subplots for different regions
- Correlation between same time periods across years

**How to read:**
- High values (>0.95): Stable annual patterns
- Lower values (<0.90): Changing consumption patterns

**What to look for:**
- **Consistent patterns**: High correlations across all year pairs
- **Trends**: Gradually decreasing correlations over time
- **Anomalies**: Specific year pairs with low correlation

**Insights:**
```
High YoY correlation (>0.95):
- Stable demand patterns
- Good predictability
- Historical data useful

Low YoY correlation (<0.90):
- Structural changes (efficiency, population)
- Economic shifts
- Weather anomalies
```

**Action items:**
- ✅ High stability = use more historical data
- ⚠️ Low stability = focus on recent data, detect drift

---

### 4. Hourly Patterns (`hourly_patterns.png`)

**What it shows:**
- Average consumption by hour of day (0-23)
- Error bands showing ±1 standard deviation

**How to read:**
- **Blue line**: Mean consumption at each hour
- **Shaded area**: Variability/uncertainty
- **Peak hours**: Highest consumption times

**What to look for:**
- **Morning peak** (7-9 AM): Work/school start
- **Afternoon peak** (1-3 PM): Maximum load
- **Evening peak** (6-8 PM): Residential return
- **Overnight low** (2-5 AM): Minimal usage

**Insights:**
```
Typical pattern:
02:00 - Minimum (~18,000 MW)
14:00 - Maximum (~35,000 MW)
Range - ~95% increase from min to max

Wide error bands indicate:
- High variability at that hour
- Seasonality effects
- Weather sensitivity
```

**Action items:**
- Model needs to capture 24-hour cycle
- Hour feature is critical
- Consider separate models for peak vs off-peak

---

### 5. Daily Patterns (`daily_patterns.png`)

**What it shows:**
- Average consumption by day of week
- Error bars showing standard deviation

**How to read:**
- **Monday (0)** through **Sunday (6)**
- Bar height = average consumption
- Error bars = day-to-day variability

**What to look for:**
- **Weekday pattern**: Mon-Fri similar, higher
- **Weekend drop**: Sat-Sun lower consumption
- **Monday effect**: Sometimes higher (week start)

**Insights:**
```
Typical pattern:
Weekdays (Mon-Fri): ~31,000 MW
Weekends (Sat-Sun): ~28,000 MW
Drop: ~10% on weekends

Small error bars = consistent pattern
Large error bars = high variability
```

**Action items:**
- Include day-of-week feature
- Consider weekday/weekend binary feature
- Model can learn this pattern automatically

---

### 6. Seasonal Patterns (`seasonal_patterns.png`)

**What it shows:**
- Average consumption by month (1-12)
- Shaded area showing ±1 standard deviation

**How to read:**
- **X-axis**: Months (Jan-Dec)
- **Green line**: Mean monthly consumption
- **Shaded green**: Variability within month

**What to look for:**
- **Summer peak** (July-August): AC load
- **Winter peak** (December-February): Heating
- **Spring/Fall valleys**: Moderate weather

**Insights:**
```
Typical annual pattern:
Summer peak (July): ~35,000 MW
Winter peak (Jan): ~33,000 MW
Spring low (April): ~27,000 MW

Wide bands in summer = weather variability
Narrow bands in spring = stable conditions
```

**Action items:**
- Strong seasonality requires month/quarter features
- May need separate models per season
- Consider external weather data

---

### 7. Decomposition (`decomposition.png`)

**What it shows:**
- Time series broken into components:
  - **Observed**: Original data
  - **Trend**: Long-term direction
  - **Seasonal**: Repeating patterns
  - **Residual**: Random noise

**How to read:**
- Top to bottom: Observed → Trend → Seasonal → Residual
- Each plot shows one component over time

**What to look for:**
- **Trend**: Upward/downward/stable
- **Seasonal**: Clear 365-day pattern
- **Residual**: Should look random

**Insights:**
```
Good decomposition:
- Trend is smooth
- Seasonal shows clear annual cycle
- Residual looks like white noise

Poor decomposition:
- Residual has patterns (missed structure)
- Seasonal irregular (non-stationary)
```

**Action items:**
- Upward trend = include time features
- Strong seasonal = model needs to capture year cycle
- Patterned residuals = missing features

---

## Model Performance Figures

### 8. Training History (`lstm_training_history.png` or `gru_training_history.png`)

**What it shows:**
- Training and validation loss over epochs
- Model learning progression

**How to read:**
- **X-axis**: Training epochs
- **Blue line with circles**: Training loss
- **Orange line with squares**: Validation loss

**What to look for:**
- **Both decreasing**: Good learning
- **Converging**: Model is fitting well
- **Diverging**: Overfitting

**Healthy patterns:**
```
Good training:
Epoch   1: Train: 0.13  Val: 0.08  ← Val better (normal at start)
Epoch  10: Train: 0.04  Val: 0.05  ← Converging
Epoch  20: Train: 0.02  Val: 0.03  ← Close together (good!)
Epoch  30: Early stopping            ← Optimal point

Problems to watch for:
- Val loss increasing → Overfitting
- Val loss not decreasing → Underfitting
- Large gap → Need regularization
```

**Action items:**
- ✅ **Small gap (<20%)**: Good model
- ⚠️ **Gap widening**: Increase dropout, reduce capacity
- ⚠️ **Both high**: Increase capacity, train longer
- ⚠️ **Val not improving**: Adjust learning rate

---

### 9. Model Predictions (`lstm_predictions.png` or `gru_predictions.png`)

**What it shows:**
- **Top panel**: Time series of predictions vs actual
- **Bottom panel**: Scatter plot of predicted vs actual

**Top Panel - Time Series:**

**How to read:**
- **Blue line**: Actual values
- **Orange line**: Model predictions
- Last 500 time steps shown

**What to look for:**
- **Overlap**: Predictions following actual closely
- **Phase lag**: Predictions delayed = copying lag feature
- **Amplitude**: Predictions capturing peaks/valleys

**Quality indicators:**
```
Excellent: Lines nearly overlap
Good: Predictions follow major patterns
Fair: General trend captured, details missed
Poor: Large deviations, phase lag
```

**Bottom Panel - Scatter Plot:**

**How to read:**
- **Dots**: Each point is one prediction
- **Red dashed line**: Perfect prediction (y=x)
- **Tightness to line**: Prediction accuracy

**What to look for:**
- **Tight cluster**: High accuracy
- **Spread**: Prediction uncertainty
- **Systematic bias**: Points above/below line

**Quality indicators:**
```
Excellent: R² > 0.95, tight clustering
Good: R² > 0.90, moderate spread
Fair: R² > 0.85, noticeable spread
Poor: R² < 0.85, wide scatter

Metrics shown:
RMSE: Lower is better (MW error)
MAE: Average absolute error
MAPE: Percentage error (aim for <5%)
```

**Action items:**
- ✅ **Good overlap**: Model is working well
- ⚠️ **Phase lag**: Reduce lag feature weight
- ⚠️ **Missing peaks**: Increase model capacity
- ⚠️ **Systematic bias**: Check data preprocessing

---

### 10. Shallow Model Predictions (`shallow_predictions.png`)

**What it shows:**
- 4 subplots for Linear, Ridge, Lasso, Random Forest
- Actual vs predicted for last 500 time steps

**How to read:**
- Each subplot shows time series comparison
- RMSE and R² metrics in title

**What to look for:**
- Model comparison at a glance
- Which simple model performs best
- Baseline for deep learning models

**Typical results:**
```
Linear Regression:
- Smooth predictions
- Misses non-linear patterns
- RMSE: ~1500 MW

Random Forest:
- Captures non-linearity
- Some overfitting visible
- RMSE: ~1100 MW

Compare to LSTM/GRU:
- Should show improvement
- Better peak capture
- RMSE: ~800-1000 MW
```

**Action items:**
- Use best traditional model as baseline
- If deep learning isn't better, investigate why
- Consider ensemble of traditional + deep learning

---

### 11. Shallow Model Scatter (`shallow_scatter.png`)

**What it shows:**
- Scatter plots for 4 traditional models
- Predicted vs actual values

**How to read:**
- Same format as deep learning scatter
- R² score for each model

**What to look for:**
- Linearity of relationship
- Heteroscedasticity (varying spread)
- Outliers

**Common patterns:**
```
Linear models (Ridge, Lasso):
- Clear linear relationship
- Uniform spread
- Some systematic under/overprediction

Random Forest:
- Better overall fit
- Handles non-linearity
- May show discretization effects

Goal: Deep learning should show:
- Tighter clustering
- Higher R²
- Less systematic bias
```

---

## Feature Analysis Figures

### 12. Feature Importance (`feature_importance.png`)

**What it shows:**
- Top 10 most important features for XGBoost, LightGBM, CatBoost
- Horizontal bar charts

**How to read:**
- **Longer bars**: More important features
- **Consistent across models**: Robust findings
- **Model-specific**: May indicate overfitting

**What to look for:**
- **Lag features dominating**: Historical values predictive
- **Temporal features**: Hour, day, month importance
- **Rolling statistics**: Smoothing helpful

**Typical ranking:**
```
1. lag_24 (~35%): Same hour yesterday
2. rolling_mean_24 (~22%): Daily average
3. hour (~18%): Time of day
4. lag_168 (~12%): Same hour last week
5. month (~8%): Seasonal pattern
6. Others (<5%): Supporting features
```

**Action items:**
- ✅ **Lag features important**: Good data quality
- ⚠️ **Only lag_1 important**: Model just copying
- ⚠️ **Unexpected features**: Investigate data leakage
- ℹ️ **Deep learning**: Learns these automatically

---

## Diagnostic Patterns

### Pattern 1: Overfitting

**Signs:**
- Training loss << Validation loss (large gap)
- Training plot: diverging lines
- Predictions: perfect on train, poor on test

**Solutions:**
- Increase dropout (0.2 → 0.4)
- Reduce model size
- Add more training data
- Early stopping (already enabled)

### Pattern 2: Underfitting

**Signs:**
- Both losses high and not improving
- Training plot: plateaued early
- Predictions: missing major patterns

**Solutions:**
- Increase model capacity
- Train more epochs
- Reduce regularization
- Add more features

### Pattern 3: Phase Lag

**Signs:**
- Predictions delayed by 1-2 timesteps
- Time series: offset pattern
- Scatter: linear but biased

**Solutions:**
- Model learning to copy lag_1
- Reduce lag feature weight
- Increase hidden size
- Use longer lookback window

### Pattern 4: Peak Flattening

**Signs:**
- Predictions smooth out extremes
- Missing high peaks and low valleys
- Scatter: compressed vertically

**Solutions:**
- Use quantile loss instead of MSE
- Increase model capacity
- Reduce regularization
- Add peak-specific features

## Best Practices

### Before Training
1. **Check data exploration figures** - Understand patterns
2. **Verify feature importance** - Ensure good features
3. **Inspect decomposition** - Check for trends/seasonality

### During Training
1. **Monitor training curves** - Watch for overfitting
2. **Check early stopping** - Note optimal epoch
3. **Track validation loss** - Ensure improvement

### After Training
1. **Examine predictions** - Visual inspection first
2. **Check scatter plots** - Look for systematic bias
3. **Compare to baseline** - Verify improvement
4. **Review time series** - Ensure no phase lag

## Quick Diagnostics

```python
# Generate all figures
python scripts/generate_figures.py

# Check training performance
# Look at: figures/lstm_training_history.png
# Good: Converging lines, early stopping after 20-50 epochs
# Bad: Diverging lines or no improvement

# Check prediction quality
# Look at: figures/lstm_predictions.png
# Good: RMSE < 1000 MW, MAPE < 4%, tight scatter
# Bad: RMSE > 1500 MW, MAPE > 6%, wide scatter

# Compare to baseline
# Look at: figures/shallow_predictions.png
# Deep learning should beat Random Forest by 20-30%
```

## Troubleshooting

**Q: Training curves are noisy**
- A: Normal with small batch sizes, look at overall trend

**Q: Predictions lag actual by one step**
- A: Model copying lag_1 feature, increase capacity

**Q: Good training but poor test performance**
- A: Overfit or distribution shift, use more validation data

**Q: Scatter plot shows horizontal/vertical lines**
- A: Model predicting constant values, check data/features

**Q: Feature importance mostly lag features**
- A: Normal and expected, temporal features secondary

## Next Steps

After reviewing visualisations:
- If performance good → [Deploy or experiment further](LSTM_GUIDE.md)
- If overfitting → Increase regularization
- If underfitting → Increase model capacity
- If phase lag → Adjust features/architecture

See [Model Comparison](MODEL_COMPARISON.md) for choosing different models.
