"""Standalone script for training and evaluating LSTM/GRU models for time-series forecasting.

This script provides a complete pipeline for:
- Loading and preprocessing energy consumption data
- Engineering temporal and statistical features
- Training LSTM or GRU models with GPU acceleration
- Evaluating model performance on test data
- Generating training history and prediction visualisations
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")


class TimeSeriesDataset(Dataset):
    """Dataset for time-series forecasting with sliding window approach."""

    def __init__(self, data, target, lookback, forecast_horizon):
        """Initialise dataset with features and targets.

        Args:
            data: numpy array of shape (n_samples, n_features)
            target: numpy array of shape (n_samples,)
            lookback: number of timesteps to look back
            forecast_horizon: number of timesteps to forecast
        """
        self.data = torch.FloatTensor(data)
        self.target = torch.FloatTensor(target)
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.lookback - self.forecast_horizon + 1

    def __getitem__(self, idx):
        # Get sequence of features
        x = self.data[idx:idx + self.lookback]
        # Get target (next forecast_horizon values)
        y = self.target[idx + self.lookback:idx + self.lookback + self.forecast_horizon]
        return x, y


class LSTMModel(nn.Module):
    """LSTM/GRU model for time-series forecasting."""

    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon, dropout=0.2, use_gru=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gru = use_gru

        # LSTM or GRU layer
        if use_gru:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Dropout and output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        # Take the last hidden state
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class LSTMTrainer:
    """Handles training, validation, and testing of LSTM/GRU models."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.train_losses = []
        self.val_losses = []

    def _load_and_prepare_data(self):
        """Load and preprocess energy consumption data."""
        print(f"\n[1/5] Loading data from {self.config.data_path}...")
        df = pd.read_csv(self.config.data_path, parse_dates=["Datetime"])

        df_region = df[["Datetime", self.config.region]].copy()
        df_region.columns = ["Datetime", "energy"]
        df_region = df_region.dropna()
        print(f"   Loaded {len(df_region):,} rows for {self.config.region}")

        print("\n[2/5] Engineering features...")
        df_region["hour"] = df_region["Datetime"].dt.hour
        df_region["dayofweek"] = df_region["Datetime"].dt.dayofweek
        df_region["month"] = df_region["Datetime"].dt.month
        df_region["day"] = df_region["Datetime"].dt.day
        df_region["quarter"] = df_region["Datetime"].dt.quarter

        # Add lag and rolling features
        df_region['lag_1'] = df_region['energy'].shift(1)
        df_region['lag_24'] = df_region['energy'].shift(24)
        df_region['lag_168'] = df_region['energy'].shift(168)
        df_region['rolling_mean_24'] = df_region['energy'].rolling(window=24).mean()
        df_region['rolling_std_24'] = df_region['energy'].rolling(window=24).std()

        df_region = df_region.dropna().reset_index(drop=True)
        print(f"   Final dataset: {len(df_region):,} rows")

        return df_region

    def _create_datasets(self, df):
        """Creates training, validation, and test datasets."""
        print("\n[3/5] Creating datasets...")

        # Select features
        feature_cols = ['hour', 'dayofweek', 'month', 'day', 'quarter',
                       'lag_1', 'lag_24', 'lag_168', 'rolling_mean_24', 'rolling_std_24']
        target_col = 'energy'

        X = df[feature_cols].values
        y = df[target_col].values.reshape(-1, 1)

        # Train/Val/Test split
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # Normalize features and target
        X_train = self.scaler_X.fit_transform(X_train)
        X_val = self.scaler_X.transform(X_val)
        X_test = self.scaler_X.transform(X_test)

        y_train = self.scaler_y.fit_transform(y_train).flatten()
        y_val = self.scaler_y.transform(y_val).flatten()
        y_test = self.scaler_y.transform(y_test).flatten()

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train,
                                         self.config.lookback,
                                         self.config.forecast_horizon)
        val_dataset = TimeSeriesDataset(X_val, y_val,
                                       self.config.lookback,
                                       self.config.forecast_horizon)
        test_dataset = TimeSeriesDataset(X_test, y_test,
                                        self.config.lookback,
                                        self.config.forecast_horizon)

        print(f"   Train samples: {len(train_dataset):,}")
        print(f"   Val samples: {len(val_dataset):,}")
        print(f"   Test samples: {len(test_dataset):,}")

        return train_dataset, val_dataset, test_dataset

    def _build_model(self, input_size):
        """Builds the LSTM/GRU model."""
        model_type = "GRU" if self.config.use_gru else "LSTM"
        print(f"\n[4/5] Creating {model_type} model...")

        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            forecast_horizon=self.config.forecast_horizon,
            dropout=self.config.dropout,
            use_gru=self.config.use_gru
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Model created: {total_params:,} parameters")

    def train(self):
        """Trains the model."""
        print(f"\n{'='*80}")
        print(f"Training {'GRU' if self.config.use_gru else 'LSTM'} Model")
        print(f"{'='*80}")
        print(f"Device: {self.device}")

        df = self._load_and_prepare_data()
        train_dataset, val_dataset, test_dataset = self._create_datasets(df)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size,
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size,
                               shuffle=False, num_workers=0)

        # Build model
        input_size = train_dataset.data.shape[1]
        self._build_model(input_size)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        print("\n[5/5] Training model...")
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    output = self.model(batch_X)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            # Store losses for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"   Epoch {epoch+1:3d}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_path = Path("checkpoints") / f"lstm_best_{self.config.region}.pt"
                checkpoint_path.parent.mkdir(exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'config': self.config,
                    'scaler_X': self.scaler_X,
                    'scaler_y': self.scaler_y,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"\n   Early stopping triggered after {epoch+1} epochs")
                    break

        print(f"\n   Training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Model saved to: {checkpoint_path}")

        # Plot training history
        self._plot_training_history(checkpoint_path)

        return checkpoint_path, test_dataset

    def _plot_training_history(self, checkpoint_path):
        """Plots and saves training history."""
        figures_dir = Path("figures")
        figures_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs_range = range(1, len(self.train_losses) + 1)
        ax.plot(epochs_range, self.train_losses, label='Train Loss', linewidth=2, marker='o')
        ax.plot(epochs_range, self.val_losses, label='Val Loss', linewidth=2, marker='s')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        ax.set_title(f'{"GRU" if self.config.use_gru else "LSTM"} Training History',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        model_type = "gru" if self.config.use_gru else "lstm"
        save_path = figures_dir / f"{model_type}_training_history.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"   Training history plot saved to: {save_path}")

    def test(self, checkpoint_path, test_dataset=None):
        """Tests the trained model."""
        print(f"\n{'='*80}")
        print("Testing Model")
        print(f"{'='*80}")

        if test_dataset is None:
            df = self._load_and_prepare_data()
            _, _, test_dataset = self._create_datasets(df)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Build model with same architecture
        input_size = test_dataset.data.shape[1]
        self._build_model(input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']

        # Test
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size,
                                shuffle=False, num_workers=0)

        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                output = self.model(batch_X)
                all_preds.append(output.cpu().numpy())
                all_targets.append(batch_y.numpy())

        # Concatenate and inverse transform
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Inverse transform to original scale
        preds_original = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
        targets_original = self.scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()

        # Calculate metrics
        mse = np.mean((preds_original - targets_original) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds_original - targets_original))
        mape = np.mean(np.abs((targets_original - preds_original) / targets_original)) * 100

        print("\n   Test Results:")
        print(f"   MSE:  {mse:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAE:  {mae:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print("\n   Testing complete!")

        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'predictions': preds_original,
            'targets': targets_original
        }

        # Plot predictions
        self._plot_predictions(results)

        return results

    def _plot_predictions(self, results):
        """Plots and saves prediction results."""
        figures_dir = Path("figures")
        figures_dir.mkdir(exist_ok=True)

        model_type = "gru" if self.config.use_gru else "lstm"
        preds = results['predictions']
        targets = results['targets']

        # Time series plot (last 500 points for clarity)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot last 500 predictions
        n_show = min(500, len(preds))
        time_steps = range(n_show)
        ax1.plot(time_steps, targets[-n_show:], label='Actual', linewidth=2, alpha=0.7)
        ax1.plot(time_steps, preds[-n_show:], label='Predicted', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Energy (MW)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{model_type.upper()} Predictions vs Actual (Last {n_show} points)',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Scatter plot
        ax2.scatter(targets, preds, alpha=0.3, s=10)
        min_val, max_val = min(targets.min(), preds.min()), max(targets.max(), preds.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Energy (MW)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Predicted Energy (MW)', fontsize=12, fontweight='bold')
        ax2.set_title(f'{model_type.upper()} Actual vs Predicted\n'
                     f'RMSE: {results["rmse"]:.2f}, MAE: {results["mae"]:.2f}, MAPE: {results["mape"]:.2f}%',
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = figures_dir / f"{model_type}_predictions.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"   Predictions plot saved to: {save_path}")


def main():
    """Main function to run the training and testing pipeline."""
    parser = argparse.ArgumentParser(description="Train/Test LSTM/GRU for energy forecasting")

    # General arguments
    parser.add_argument("--mode", type=str, default="train_test",
                       choices=["train", "test", "train_test"],
                       help="Mode of operation")
    parser.add_argument("--data_path", type=str, default="composite_energy_data.csv",
                       help="Path to the dataset")
    parser.add_argument("--region", type=str, default="PJME_MW",
                       help="Energy region to forecast")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to model checkpoint for testing")

    # Model architecture
    parser.add_argument("--use_gru", action="store_true",
                       help="Use GRU instead of LSTM")
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="Hidden size for LSTM/GRU")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of LSTM/GRU layers")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")

    # Time series parameters
    parser.add_argument("--lookback", type=int, default=168,
                       help="Lookback window size (hours)")
    parser.add_argument("--forecast_horizon", type=int, default=24,
                       help="Forecast horizon (hours)")

    config = parser.parse_args()

    if not Path(config.data_path).exists():
        print(f"\n❌ ERROR: Data file not found: {config.data_path}")
        print("   Please run the exploration.ipynb notebook first to generate data.")
        return

    trainer = LSTMTrainer(config)

    if config.mode in ["train", "train_test"]:
        checkpoint_path, test_dataset = trainer.train()

        if config.mode == "train_test":
            trainer.test(checkpoint_path, test_dataset)

    elif config.mode == "test":
        if not config.checkpoint_path:
            print("\n❌ ERROR: No checkpoint specified for testing.")
            print("   Provide a path via --checkpoint_path.")
            return
        trainer.test(config.checkpoint_path)

    print("\n" + "="*80)
    print("Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
