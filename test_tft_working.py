"""Test script using lightning.pytorch instead of pytorch_lightning"""
import sys
print(f"Python executable: {sys.executable}")

# Check package versions
import lightning
import pytorch_forecasting as pf
import torch

print(f"\nPyTorch: {torch.__version__}")
print(f"Lightning: {lightning.__version__}")
print(f"PyTorch Forecasting: {pf.__version__}")

# Check if TFT can be imported and used
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
import pandas as pd
import numpy as np

print("\n[OK] All imports successful")

# Create minimal test dataset
np.random.seed(42)
data = pd.DataFrame({
    'time_idx': np.repeat(np.arange(100), 1),
    'series_id': '1',
    'value': np.random.randn(100).cumsum(),
})

# Create TimeSeriesDataSet
training = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="value",
    group_ids=["series_id"],
    max_encoder_length=24,
    max_prediction_length=6,
    time_varying_unknown_reals=["value"],
)

print("[OK] TimeSeriesDataSet created successfully")

# Create TFT model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
)

print("[OK] TFT model created successfully")

# IMPORTANT: Use lightning.pytorch.Trainer instead of pytorch_lightning.Trainer
from lightning.pytorch import Trainer

trainer = Trainer(
    max_epochs=1,
    accelerator="cpu",
    devices=1,
    enable_progress_bar=False,
    enable_model_summary=False,
    logger=False,
)

print("\n[OK] Trainer created successfully")

# This is where the error was occurring
try:
    train_dataloader = training.to_dataloader(batch_size=4)
    trainer.fit(tft, train_dataloaders=train_dataloader)
    print("\n=== SUCCESS! trainer.fit() completed without errors ===")
except TypeError as e:
    print(f"\n=== ERROR: {e} ===")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
