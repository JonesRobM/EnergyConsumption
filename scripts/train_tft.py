"""
Standalone script to train and test Temporal Fusion Transformer (TFT)
for energy consumption forecasting.

Usage:
    python train_tft.py --mode train          # Train new model
    python train_tft.py --mode test           # Test existing model
    python train_tft.py --mode train_test     # Train and test
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import PyTorch Forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# Import Lightning (new namespace)
try:
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    # Fallback to old namespace
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

print("=" * 80)
print("TFT Energy Consumption Forecasting - Training & Testing Script")
print("=" * 80)


def load_and_prepare_data(data_path='composite_energy_data.csv', region='PJME_MW'):
    """Load and prepare energy consumption data for TFT."""

    print(f"\n[1/6] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Filter to single region
    df_region = df[['Datetime', region]].copy()
    df_region.columns = ['Datetime', 'energy']
    df_region = df_region.dropna()

    print(f"   Loaded {len(df_region):,} rows for {region}")

    # Extract temporal features
    print("\n[2/6] Engineering features...")
    df_region['hour'] = df_region['Datetime'].dt.hour
    df_region['dayofweek'] = df_region['Datetime'].dt.dayofweek
    df_region['month'] = df_region['Datetime'].dt.month
    df_region['year'] = df_region['Datetime'].dt.year

    # Convert temporal features to string for categorical handling
    df_region['hour'] = df_region['hour'].astype(str)
    df_region['dayofweek'] = df_region['dayofweek'].astype(str)
    df_region['month'] = df_region['month'].astype(str)

    # Cyclical encodings
    df_region['hour_sin'] = np.sin(2 * np.pi * df_region['hour'].astype(int) / 24)
    df_region['hour_cos'] = np.cos(2 * np.pi * df_region['hour'].astype(int) / 24)
    df_region['month_sin'] = np.sin(2 * np.pi * df_region['month'].astype(int) / 12)
    df_region['month_cos'] = np.cos(2 * np.pi * df_region['month'].astype(int) / 12)
    df_region['dayofweek_sin'] = np.sin(2 * np.pi * df_region['dayofweek'].astype(int) / 7)
    df_region['dayofweek_cos'] = np.cos(2 * np.pi * df_region['dayofweek'].astype(int) / 7)

    # Lag features
    df_region['lag_24'] = df_region['energy'].shift(24)
    df_region['lag_168'] = df_region['energy'].shift(168)

    # Rolling features
    df_region['rolling_mean_24'] = df_region['energy'].rolling(24, min_periods=1).mean()
    df_region['rolling_mean_168'] = df_region['energy'].rolling(168, min_periods=1).mean()

    # Create time index (sequential hours)
    df_region = df_region.sort_values('Datetime').reset_index(drop=True)
    df_region['time_idx'] = df_region.index

    # Series ID (required for TimeSeriesDataSet)
    df_region['series_id'] = region

    # Remove initial NaN rows from lag features
    df_region = df_region.dropna().reset_index(drop=True)
    df_region['time_idx'] = df_region.index

    print(f"   Features created: {len(df_region.columns)} columns")
    print(f"   Final dataset: {len(df_region):,} rows")

    return df_region


def create_datasets(df, encoder_length=168, prediction_length=24,
                    train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test TimeSeriesDataSets."""

    print(f"\n[3/6] Creating TimeSeriesDataSets...")
    print(f"   Encoder length: {encoder_length} hours (lookback)")
    print(f"   Prediction length: {prediction_length} hours (forecast)")

    # Calculate split points
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    print(f"   Train: 0 to {train_end:,} ({train_ratio*100:.0f}%)")
    print(f"   Val:   {train_end:,} to {val_end:,} ({val_ratio*100:.0f}%)")
    print(f"   Test:  {val_end:,} to {n:,} ({(1-train_ratio-val_ratio)*100:.0f}%)")

    # Training dataset
    training = TimeSeriesDataSet(
        df.iloc[:train_end],
        time_idx="time_idx",
        target="energy",
        group_ids=["series_id"],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,

        # Static features (constant for all timesteps)
        static_categoricals=["series_id"],

        # Time-varying known features (available for future)
        time_varying_known_categoricals=["hour", "dayofweek", "month"],
        time_varying_known_reals=[
            "time_idx", "hour_sin", "hour_cos",
            "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"
        ],

        # Time-varying unknown features (only available until present)
        time_varying_unknown_reals=[
            "energy", "lag_24", "lag_168",
            "rolling_mean_24", "rolling_mean_168"
        ],

        # Normalization
        target_normalizer=GroupNormalizer(
            groups=["series_id"], transformation="softplus"
        ),

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Validation dataset (from training parameters)
    validation = TimeSeriesDataSet.from_dataset(
        training, df.iloc[train_end:val_end], predict=False, stop_randomization=True
    )

    # Test dataset (from training parameters)
    test = TimeSeriesDataSet.from_dataset(
        training, df.iloc[val_end:], predict=False, stop_randomization=True
    )

    print(f"   ✓ Datasets created successfully")

    return training, validation, test


def create_model(training_dataset, hidden_size=64, attention_heads=4,
                 dropout=0.1, learning_rate=0.001):
    """Create TFT model from dataset."""

    print(f"\n[4/6] Creating TFT model...")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Attention heads: {attention_heads}")
    print(f"   Dropout: {dropout}")
    print(f"   Learning rate: {learning_rate}")

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_heads,
        dropout=dropout,
        hidden_continuous_size=hidden_size // 2,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    total_params = sum(p.numel() for p in tft.parameters())
    trainable_params = sum(p.numel() for p in tft.parameters() if p.requires_grad)

    print(f"   ✓ Model created: {total_params:,} parameters ({trainable_params:,} trainable)")

    return tft


def train_model(tft, training, validation, max_epochs=50, batch_size=64):
    """Train TFT model with PyTorch Lightning."""

    print(f"\n[5/6] Training model...")
    print(f"   Max epochs: {max_epochs}")
    print(f"   Batch size: {batch_size}")

    # Create dataloaders
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint = ModelCheckpoint(
        dirpath='checkpoints',
        filename='tft-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
    )

    # Logger
    logger = TensorBoardLogger("lightning_logs", name="tft_energy")

    # Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop, lr_monitor, checkpoint],
        logger=logger,
        enable_progress_bar=True,
    )

    # Train
    print("\n   Starting training...")
    print("   " + "-" * 60)
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print("\n   ✓ Training complete!")
    print(f"   Best model: {checkpoint.best_model_path}")

    return trainer, checkpoint.best_model_path


def test_model(model_path, test_dataset, training_dataset):
    """Test TFT model on test set."""

    print(f"\n[6/6] Testing model...")
    print(f"   Loading from: {model_path}")

    # Hotfix for PyTorch 2.6+ unpickling security change
    # see: https://github.com/Lightning-AI/pytorch-lightning/issues/20261
    torch.serialization.add_safe_globals([GroupNormalizer, pd.DataFrame])

    # Load best model
    best_tft = TemporalFusionTransformer.load_from_checkpoint(model_path)

    # Create test dataloader
    test_dataloader = test_dataset.to_dataloader(
        train=False, batch_size=128, num_workers=0
    )

    # Evaluate
    trainer = Trainer(accelerator="auto", devices=1)
    test_results = trainer.test(best_tft, dataloaders=test_dataloader, verbose=False)

    print("\n   Test Results:")
    print("   " + "-" * 60)
    for key, value in test_results[0].items():
        print(f"   {key:30s}: {value:.4f}")

    # Make predictions on a few examples
    print("\n   Making sample predictions...")
    predictions = best_tft.predict(
        test_dataloader, mode="prediction", return_x=True, trainer_kwargs=dict(accelerator="auto")
    )

    print(f"   ✓ Generated {len(predictions.output)} predictions")

    return test_results, predictions


def main():
    parser = argparse.ArgumentParser(description='Train/Test TFT for energy forecasting')
    parser.add_argument('--mode', type=str, default='train_test',
                        choices=['train', 'test', 'train_test'],
                        help='Mode: train, test, or train_test')
    parser.add_argument('--data', type=str, default='composite_energy_data.csv',
                        help='Path to data CSV')
    parser.add_argument('--region', type=str, default='PJME_MW',
                        help='Energy region to forecast')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size')
    parser.add_argument('--attention_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for testing')

    args = parser.parse_args()

    # Check if data exists
    if not Path(args.data).exists():
        print(f"\n❌ ERROR: Data file not found: {args.data}")
        print("   Please run the exploration.ipynb notebook first to generate data.")
        return

    # Load data
    df = load_and_prepare_data(args.data, args.region)

    # Create datasets
    training, validation, test = create_datasets(df)

    if args.mode in ['train', 'train_test']:
        # Create and train model
        tft = create_model(
            training,
            hidden_size=args.hidden_size,
            attention_heads=args.attention_heads
        )

        trainer, best_model_path = train_model(
            tft, training, validation,
            max_epochs=args.epochs,
            batch_size=args.batch_size
        )

        # Save path for testing
        checkpoint_path = best_model_path

    if args.mode in ['test', 'train_test']:
        # Determine checkpoint path
        if args.mode == 'test':
            if args.checkpoint is None:
                print("\n❌ ERROR: --checkpoint required for test mode")
                return
            checkpoint_path = args.checkpoint

        # Test model
        test_results, predictions = test_model(checkpoint_path, test, training)

    print("\n" + "=" * 80)
    print("✓ Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
