"Standalone script for training and evaluating the Temporal Fusion Transformer."
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import (
    GroupNormalizer,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.metrics import QuantileLoss

# Suppress warnings
warnings.filterwarnings("ignore")

# Set float precision for matrix multiplication
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


class TFTModel:
    """
    Encapsulates the entire TFT pipeline from data loading to testing.
    """

    def __init__(self, config):
        self.config = config
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.model = None
        self.trainer = None

    def _load_and_prepare_data(self):
        """Loads and preprocesses the energy consumption data."""
        print(f"\n[1/6] Loading data from {self.config.data_path}...")
        df = pd.read_csv(self.config.data_path, parse_dates=["Datetime"])

        df_region = df[["Datetime", self.config.region]].copy()
        df_region.columns = ["Datetime", "energy"]
        df_region = df_region.dropna()
        print(f"   Loaded {len(df_region):,} rows for {self.config.region}")

        print("\n[2/6] Engineering features...")
        df_region["hour"] = df_region["Datetime"].dt.hour.astype(str)
        df_region["dayofweek"] = df_region["Datetime"].dt.dayofweek.astype(str)
        df_region["month"] = df_region["Datetime"].dt.month.astype(str)
        df_region["time_idx"] = (
            (df_region["Datetime"] - df_region["Datetime"].min()).dt.total_seconds() // 3600
        ).astype(int)
        df_region["series_id"] = self.config.region
        
        # Simple lag and rolling mean features
        df_region['lag_24'] = df_region['energy'].shift(24)
        df_region['rolling_mean_24'] = df_region['energy'].rolling(window=24).mean()
        df_region = df_region.dropna().reset_index(drop=True)

        print(f"   Final dataset: {len(df_region):,} rows")
        return df_region

    def _create_datasets(self, df):
        """Creates training, validation, and test datasets."""
        print("\n[3/6] Creating TimeSeriesDataSets...")
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        self.training_data = TimeSeriesDataSet(
            df.iloc[:train_end],
            time_idx="time_idx",
            target="energy",
            group_ids=["series_id"],
            max_encoder_length=self.config.encoder_length,
            max_prediction_length=self.config.prediction_length,
            static_categoricals=["series_id"],
            time_varying_known_categoricals=["hour", "dayofweek", "month"],
            time_varying_unknown_reals=["energy", "lag_24", "rolling_mean_24"],
            target_normaliser=GroupNormalizer(
                groups=["series_id"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        self.validation_data = TimeSeriesDataSet.from_dataset(
            self.training_data, df.iloc[train_end:val_end], predict=False, stop_randomization=True
        )
        self.test_data = TimeSeriesDataSet.from_dataset(
            self.training_data, df.iloc[val_end:], predict=False, stop_randomization=True
        )
        print("   ✓ Datasets created successfully")

    def _build_model(self):
        """Builds the TFT model."""
        print("\n[4/6] Creating TFT model...")
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_data,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_heads,
            dropout=self.config.dropout,
            hidden_continuous_size=self.config.hidden_size // 2,
            loss=QuantileLoss(),
            log_interval=-1,
            reduce_on_plateau_patience=4,
        )
        print(f"   ✓ Model created: {self.model.size()} parameters")

    def train(self):
        """Trains the model."""
        print("\n[5/6] Training model...")
        df = self._load_and_prepare_data()
        self._create_datasets(df)
        self._build_model()

        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min"
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="tft-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        logger = TensorBoardLogger("lightning_logs", name="tft_energy_v2")

        self.trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator="auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=logger,
        )

        self.trainer.fit(
            self.model,
            train_dataloaders=self.training_data.to_dataloader(
                train=True, batch_size=self.config.batch_size, num_workers=0
            ),
            val_dataloaders=self.validation_data.to_dataloader(
                train=False, batch_size=self.config.batch_size * 10, num_workers=0
            ),
        )
        
        print("\n   ✓ Training complete!")
        print(f"   Best model saved at: {checkpoint_callback.best_model_path}")
        return checkpoint_callback.best_model_path
        
    def test(self, checkpoint_path=None):
        """Tests the trained model."""
        print("\n[6/6] Testing model...")
        if not checkpoint_path:
            raise ValueError("Checkpoint path must be provided for testing.")
            
        print(f"   Loading from: {checkpoint_path}")
        
        # ** KEY CHANGE: Load checkpoint manually to bypass unpickling errors **
        # This is the recommended approach when facing security-related unpickling issues
        # with versions of PyTorch that default to weights_only=True.
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
        
        # Re-create the model architecture from the training data
        # (which must be available or re-created)
        df = self._load_and_prepare_data()
        self._create_datasets(df) # This re-initialises self.training_data
        
        loaded_model = TemporalFusionTransformer.from_dataset(self.training_data)
        
        # Load the trained weights into the new model instance
        loaded_model.load_state_dict(checkpoint['state_dict'])

        test_dataloader = self.test_data.to_dataloader(
            train=False, batch_size=self.config.batch_size * 10, num_workers=0
        )
        
        # Evaluate using a new Trainer instance
        test_trainer = pl.Trainer(accelerator="auto", logger=False)
        test_results = test_trainer.test(loaded_model, dataloaders=test_dataloader, verbose=False)

        print("\n   Test Results:")
        for key, value in test_results[0].items():
            print(f"   {key:20s}: {value:.4f}")
        
        print("   ✓ Testing complete!")
        return test_results


def main():
    """Main function to run the training and testing pipeline."""
    parser = argparse.ArgumentParser(description="Train/Test TFT for energy forecasting")
    
    # General arguments
    parser.add_argument("--mode", type=str, default="train_test", choices=["train", "test", "train_test"], help="Mode of operation")
    parser.add_argument("--data_path", type=str, default="composite_energy_data.csv", help="Path to the dataset")
    parser.add_argument("--region", type=str, default="PJME_MW", help="Energy region to forecast")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint for testing")
    
    # Model hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=64, help="TFT hidden layer size")
    parser.add_argument("--attention_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Time series parameters
    parser.add_argument("--encoder_length", type=int, default=168, help="Lookback window size (hours)")
    parser.add_argument("--prediction_length", type=int, default=24, help="Forecast horizon (hours)")

    config = parser.parse_args()
    
    if not Path(config.data_path).exists():
        print(f"\n❌ ERROR: Data file not found: {config.data_path}")
        print("   Please run the exploration.ipynb notebook first to generate data.")
        return

    model_pipeline = TFTModel(config)
    best_model_path = None

    if "train" in config.mode:
        best_model_path = model_pipeline.train()
    
    if "test" in config.mode:
        # Use the newly trained model if available, otherwise use specified checkpoint
        test_checkpoint = best_model_path if best_model_path else config.checkpoint_path
        if not test_checkpoint:
            print("\n❌ ERROR: No checkpoint specified for testing.")
            print("   Either train a model first or provide a path via --checkpoint_path.")
            return
        model_pipeline.test(checkpoint_path=test_checkpoint)

    print("\n" + "="*80)
    print("✓ Complete!")
    print("="*80)

if __name__ == "__main__":
    main()