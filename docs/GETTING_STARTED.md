# Getting Started

Quick start guide to get you up and running with energy consumption forecasting.

## Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (optional but recommended)
- Kaggle API credentials for data download

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd EnergyConsumption
```

### 2. Set Up Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Place `kaggle.json` in:
   - Windows: `C:\Users\<username>\.kaggle\`
   - Linux/Mac: `~/.kaggle/`

## Quick Start: Train Your First Model

### Step 1: Prepare Data

Run the notebook to download and prepare data:

```bash
jupyter notebook notebooks/exploration.ipynb
```

Execute the first few cells to create `composite_energy_data.csv`.

### Step 2: Train an LSTM Model

**Single-line command (easiest):**
```bash
python scripts/train_lstm.py --mode train_test --epochs 50
```

**PowerShell (multi-line):**
```powershell
python scripts/train_lstm.py `
    --mode train_test `
    --epochs 50 `
    --batch_size 64
```

**Expected output:**
```
================================================================================
Training LSTM Model
================================================================================
Device: cuda

[1/5] Loading data from composite_energy_data.csv...
   Loaded 145,366 rows for PJME_MW

[2/5] Engineering features...
   Final dataset: 145,198 rows

[3/5] Creating datasets...
   Train samples: 101,447
   Val samples: 21,589
   Test samples: 21,589

[4/5] Creating LSTM model...
   Model created: 206,872 parameters

[5/5] Training model...
   Epoch   1/50 | Train Loss: 0.1360 | Val Loss: 0.0775
   ...
```

### Step 3: Check Results

After training completes, check:
- **Model checkpoint**: `checkpoints/lstm_best_PJME_MW.pt`
- **Training plot**: `figures/lstm_training_history.png`
- **Predictions plot**: `figures/lstm_predictions.png`
- **Test metrics**: Shown in terminal output

## Understanding the Results

**Key Metrics:**
- **RMSE** (Root Mean Squared Error): Lower is better, measured in MW
- **MAE** (Mean Absolute Error): Average prediction error in MW
- **MAPE** (Mean Absolute Percentage Error): Percentage error, lower is better

**Target Performance:**
- RMSE: 800-1200 MW (excellent)
- RMSE: 1200-1500 MW (good)
- RMSE: 1500+ MW (needs tuning)

## Next Steps

### Option 1: Improve Your Model
See the [LSTM/GRU Guide](LSTM_GUIDE.md) for hyperparameter tuning:
- Increase model size
- Try GRU instead of LSTM
- Adjust lookback window
- Fine-tune learning rate

### Option 2: Try Different Models
- **Better accuracy**: [TFT Guide](TFT_GUIDE.md)
- **Faster training**: XGBoost/LightGBM (see exploration notebook)

### Option 3: Explore the Data
```bash
jupyter notebook notebooks/exploration.ipynb
```

### Option 4: Generate All Figures
```bash
python scripts/generate_figures.py
```

This creates comprehensive visualisations for:
- Data exploration (correlations, patterns, seasonality)
- Model performance comparisons
- Feature importance analysis

## Common Issues

### CUDA Not Available

**Problem:**
```
Device: cpu
```

**Solution:**
Your PyTorch installation doesn't have CUDA support. Install CUDA-enabled PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Data File Not Found

**Problem:**
```
ERROR: Data file not found: composite_energy_data.csv
```

**Solution:**
Run the data preparation cells in `notebooks/exploration.ipynb` first.

### Out of Memory

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
Reduce batch size:
```bash
python scripts/train_lstm.py --batch_size 16 --epochs 50
```

## Command Reference

### Training Commands

**Basic training:**
```bash
python scripts/train_lstm.py --mode train_test --epochs 50
```

**Train with custom hyperparameters:**
```bash
python scripts/train_lstm.py --hidden_size 256 --num_layers 3 --epochs 100
```

**Train GRU instead of LSTM:**
```bash
python scripts/train_lstm.py --use_gru --epochs 50
```

**Test existing model:**
```bash
python scripts/train_lstm.py --mode test --checkpoint_path checkpoints/lstm_best_PJME_MW.pt
```

### Generation Commands

**Generate all figures:**
```bash
python scripts/generate_figures.py
```

**Start Jupyter:**
```bash
jupyter notebook
```

## Getting Help

1. Check the relevant guide:
   - [LSTM/GRU Guide](LSTM_GUIDE.md)
   - [TFT Guide](TFT_GUIDE.md)
   - [Visualisation Guide](VISUALIZATION_GUIDE.md)

2. Review the [Model Comparison](MODEL_COMPARISON.md) for choosing the right model

3. Examine the generated figures in `figures/` for diagnostic information

4. Check the detailed [Analysis](../Analysis.md) for insights

## What's Next?

Once you have a working model:
1. **Experiment** with different hyperparameters
2. **Compare** multiple model architectures
3. **Visualize** your results with generated figures
4. **Deploy** your best model for production use

See the [LSTM/GRU Guide](LSTM_GUIDE.md) for detailed tuning instructions!
