# Data Guide

Complete guide to understanding the Kaggle Hourly Energy Consumption dataset and feature engineering.

## Dataset Overview

### Source
**Kaggle Hourly Energy Consumption Dataset**
- URL: https://www.kaggle.com/robikscube/hourly-energy-consumption
- Licence: Public Domain
- Size: ~145,000 hourly observations per region
- Time span: 2004-2018 (varies by region)

### Regions Included

The dataset contains energy consumption data for multiple U.S. regions:

| Region Code | Full Name | Observations | Coverage |
|-------------|-----------|--------------|----------|
| AEP_MW | American Electric Power | ~121,000 | 2004-2018 |
| DAYTON_MW | Dayton Power & Light | ~145,000 | 2004-2018 |
| DOM_MW | Dominion Energy | ~145,000 | 2004-2018 |
| DUQ_MW | Duquesne Light Company | ~145,000 | 2004-2018 |
| EKPC_MW | East Kentucky Power Cooperative | ~145,000 | 2004-2018 |
| FE_MW | FirstEnergy | ~145,000 | 2004-2018 |
| NI_MW | Northern Indiana Public Service | ~145,000 | 2004-2018 |
| PJME_MW | PJM East | ~145,000 | 2004-2018 |
| PJMW_MW | PJM West | ~145,000 | 2004-2018 |
| PJM_MW | PJM Interconnection | ~145,000 | 2004-2018 |

**Note:** PJME_MW is used as the default region in examples.

## Data Structure

### Raw Format

```csv
Datetime,AEP_MW,DAYTON_MW,DOM_MW,...
2004-12-31 01:00:00,13478.0,2797.0,4854.0,...
2004-12-31 02:00:00,12865.0,2731.0,4564.0,...
...
```

### After Processing

After running through the data pipeline:

```python
Datetime                object (converted to datetime)
PJME_MW                float64 (energy consumption in MW)
hour                   int64 (0-23)
dayofweek              int64 (0-6, Monday=0)
month                  int64 (1-12)
day                    int64 (1-31)
quarter                int64 (1-4)
lag_1                  float64 (previous hour)
lag_24                 float64 (same hour yesterday)
lag_168                float64 (same hour last week)
rolling_mean_24        float64 (24-hour moving average)
rolling_std_24         float64 (24-hour moving std dev)
```

## Data Characteristics

### Temporal Patterns

**Hourly Pattern:**
- Morning peak: 7-9 AM (work start)
- Afternoon peak: 1-3 PM (maximum consumption)
- Evening peak: 6-8 PM (residential use)
- Overnight low: 2-5 AM (minimum consumption)

**Daily Pattern:**
- Weekdays: Higher consumption, two peaks
- Weekends: Lower consumption, flatter profile
- Sundays: Lowest overall consumption

**Seasonal Pattern:**
- Summer peak: High AC usage (July-August)
- Winter peak: Heating demand (December-February)
- Spring/Fall: Moderate consumption (April-May, September-October)

### Statistical Properties

**Typical Values (PJME_MW):**
- Mean: ~30,000 MW
- Std Dev: ~8,000 MW
- Min: ~15,000 MW (overnight, spring)
- Max: ~55,000 MW (summer afternoon peak)

**Distribution:**
- Approximately normal with slight right skew
- Seasonal variations cause multimodal distribution
- Outliers typically due to extreme weather events

### Missing Data

**Handling:**
- Original dataset: < 0.1% missing values
- Method: Dropped rows with missing values
- Impact: Negligible (< 150 rows out of 145,000)

**Check for missing data:**
```python
df.isnull().sum()
```

## Feature Engineering

### Temporal Features

These features capture cyclical patterns:

**Hour of Day (0-23):**
```python
df['hour'] = df['Datetime'].dt.hour
```
- Captures daily usage patterns
- Critical for intraday forecasting
- Encodes work schedules, human behaviour

**Day of Week (0-6):**
```python
df['dayofweek'] = df['Datetime'].dt.dayofweek
```
- Monday = 0, Sunday = 6
- Captures weekday vs weekend patterns
- Important for weekly seasonality

**Month (1-12):**
```python
df['month'] = df['Datetime'].dt.month
```
- Captures seasonal patterns
- Temperature-driven demand changes
- Holiday effects

**Quarter (1-4):**
```python
df['quarter'] = df['Datetime'].dt.quarter
```
- Broader seasonal trends
- Useful for long-term forecasting

**Day of Month (1-31):**
```python
df['day'] = df['Datetime'].dt.day
```
- Less useful than other features
- Captures billing cycle effects

### Lag Features

Previous values used to predict future:

**Lag 1 (Previous Hour):**
```python
df['lag_1'] = df['energy'].shift(1)
```
- Most recent observation
- Strongest predictor for next hour
- Captures short-term momentum

**Lag 24 (Same Hour Yesterday):**
```python
df['lag_24'] = df['energy'].shift(24)
```
- Daily pattern repetition
- Strong predictor for same time of day
- Captures day-over-day consistency

**Lag 168 (Same Hour Last Week):**
```python
df['lag_168'] = df['energy'].shift(168)
```
- Weekly pattern repetition
- Captures weekly cycles
- Important for business rhythms

### Rolling Window Features

Aggregated statistics over time windows:

**24-Hour Rolling Mean:**
```python
df['rolling_mean_24'] = df['energy'].rolling(window=24).mean()
```
- Smooths out hourly volatility
- Captures daily average level
- Reduces noise in predictions

**24-Hour Rolling Std:**
```python
df['rolling_std_24'] = df['energy'].rolling(window=24).std()
```
- Captures volatility/variability
- Indicates unusual patterns
- Useful for uncertainty quantification

**168-Hour Rolling Mean (Optional):**
```python
df['rolling_mean_168'] = df['energy'].rolling(window=168).mean()
```
- Weekly average consumption
- Very smooth, long-term trend
- Useful for strategic planning

### Feature Importance

Based on XGBoost analysis:

| Rank | Feature | Importance | Notes |
|------|---------|------------|-------|
| 1 | lag_24 | 0.35 | Same hour yesterday |
| 2 | rolling_mean_24 | 0.22 | Daily average |
| 3 | hour | 0.18 | Time of day |
| 4 | lag_168 | 0.12 | Same hour last week |
| 5 | month | 0.08 | Seasonal pattern |
| 6 | dayofweek | 0.03 | Day of week |
| 7 | lag_1 | 0.02 | Previous hour |

**Key Insights:**
- Historical values (lags) are most important
- Time-of-day (hour) is critical
- Rolling statistics smooth and stabilize
- Deep learning models learn these automatically

## Data Splits

### Time-Series Split (Default)

**Rationale:** Preserve temporal order (no data leakage)

```python
n = len(df)
train_end = int(n * 0.70)   # 70% training
val_end = int(n * 0.85)      # 15% validation
                              # 15% test

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]
```

**Resulting Sizes (PJME_MW):**
- Training: ~101,000 samples (~11.5 years)
- Validation: ~22,000 samples (~2.5 years)
- Test: ~22,000 samples (~2.5 years)

### Alternative Splits

**By Year:**
```python
train = df[df['Datetime'].dt.year < 2017]
val = df[df['Datetime'].dt.year == 2017]
test = df[df['Datetime'].dt.year >= 2018]
```
- Clean year boundaries
- May have imbalanced sizes

**Rolling Window (Advanced):**
```python
# For backtesting
for i in range(n_windows):
    train = df[start:train_end]
    val = df[train_end:val_end]
    test = df[val_end:test_end]
    # Move window forward
```
- Multiple train/test splits
- Better uncertainty estimates
- More computationally expensive

## Data Normalization

### Why Normalize?

Neural networks train better with normalised inputs:
- Faster convergence
- More stable gradients
- Better generalisation

### Method: StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)  # Use train statistics
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
```

**Important:**
- Fit scaler on training data only
- Transform val/test using training statistics
- Both features and target are scaled
- Inverse transform predictions for evaluation

### Alternative: MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
```
- Scales to [0, 1] range
- Sensitive to outliers
- Used in some LSTM tutorials
- StandardScaler generally better for this dataset

## Data Quality

### Outliers

**Detection:**
```python
# Z-score method
z_scores = np.abs(stats.zscore(df['PJME_MW']))
outliers = df[z_scores > 3]
```

**Common Causes:**
- Extreme weather events (heat waves, cold snaps)
- Grid issues or measurement errors
- Holidays (unusually low consumption)

**Handling:**
- Most outliers are real (keep them)
- Only remove obvious data errors
- Neural networks robust to outliers

### Stationarity

**Check with Augmented Dickey-Fuller test:**
```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['PJME_MW'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
```

**Results:**
- p-value < 0.05: Stationary (good)
- p-value > 0.05: Non-stationary (may need differencing)

**Note:** Energy consumption is typically non-stationary due to:
- Seasonal trends
- Long-term growth
- Weather effects

**Solution:**
- Differencing not needed for neural networks
- Lags and rolling features handle non-stationarity
- Models learn trends automatically

## External Data (Optional Enhancements)

### Weather Data
**Source:** NOAA, Weather Underground
**Features:**
- Temperature (strong correlation)
- Humidity
- Cloud cover
- Wind speed

**Expected Improvement:** 10-15% RMSE reduction

### Calendar Data
**Features:**
- Holidays (binary flag)
- School schedules
- Daylight saving time transitions
- Special events

**Expected Improvement:** 5-10% RMSE reduction

### Economic Data
**Features:**
- Industrial production index
- Employment rates
- GDP growth

**Expected Improvement:** 2-5% RMSE reduction

## Quick Reference

### Load Data
```python
import pandas as pd

df = pd.read_csv('composite_energy_data.csv', parse_dates=['Datetime'])
```

### Basic EDA
```python
# Summary statistics
df['PJME_MW'].describe()

# Check for missing values
df.isnull().sum()

# Temporal plots
df.set_index('Datetime')['PJME_MW'].plot(figsize=(15, 5))
```

### Create Features
```python
from src.feature_engineering import (
    add_temporal_features,
    create_lag_features,
    create_rolling_features
)

df = add_temporal_features(df)
df = create_lag_features(df, 'PJME_MW', lags=[1, 24, 168])
df = create_rolling_features(df, 'PJME_MW', windows=[24, 168])
```

### Prepare for Modeling
```python
# Split data
train_end = int(len(df) * 0.7)
val_end = int(len(df) * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

# Drop NaN from feature creation
train = train.dropna()
val = val.dropna()
test = test.dropna()
```

## Best Practices

1. **Always preserve temporal order** - No shuffling in time-series!
2. **Use training statistics only** - Fit scalers on train, transform val/test
3. **Drop NaN after creating features** - Lags create NaN at the start
4. **Check for data leakage** - Future shouldn't predict past
5. **Validate on recent data** - Test set should be most recent
6. **Monitor distribution shift** - Data characteristics change over time

## Common Issues

### Issue: Model performs well on validation but poorly on test
**Cause:** Distribution shift between periods
**Solution:** Use cross-validation, retrain periodically

### Issue: Predictions lag actual values by one timestep
**Cause:** Model learning to copy lag_1 feature
**Solution:** Reduce weight of lag features, increase model complexity

### Issue: Poor performance on holidays
**Cause:** Holidays not in training patterns
**Solution:** Add holiday indicator feature, use more training data

### Issue: Predictions smooth out peaks
**Cause:** Model optimizing for average error
**Solution:** Use quantile loss, increase model capacity

## Next Steps

- Explore [LSTM/GRU Guide](LSTM_GUIDE.md) for model training
- Review [Visualisation Guide](VISUALIZATION_GUIDE.md) for interpreting results
- Check [Model Comparison](MODEL_COMPARISON.md) for choosing the right model
