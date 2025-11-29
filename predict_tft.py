"""
Simple script to make predictions using a trained TFT model.

Usage:
    python predict_tft.py --checkpoint checkpoints/tft-epoch=15-val_loss=0.45.ckpt
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pytorch_forecasting import TemporalFusionTransformer
from train_tft import load_and_prepare_data, create_datasets


def make_predictions(checkpoint_path, n_samples=5, save_plots=True):
    """Load model and make predictions on test set."""

    print(f"Loading model from: {checkpoint_path}")

    # Load model
    best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

    # Load data and create datasets
    print("Loading data...")
    df = load_and_prepare_data()
    training, validation, test = create_datasets(df)

    # Create test dataloader
    test_dataloader = test.to_dataloader(train=False, batch_size=128, num_workers=0)

    # Make predictions
    print(f"Making predictions on {len(test)} test samples...")
    predictions = best_tft.predict(
        test_dataloader,
        mode="prediction",
        return_x=True,
        return_y=True,
    )

    # Extract results
    pred_output = predictions.output  # Shape: [n_samples, pred_length, n_quantiles]
    pred_x = predictions.x
    pred_y = predictions.y

    print(f"\nPrediction shape: {pred_output.shape}")
    print(f"  - {pred_output.shape[0]} samples")
    print(f"  - {pred_output.shape[1]} timesteps ahead")
    print(f"  - {pred_output.shape[2]} quantiles")

    # Plot random samples
    if save_plots:
        print(f"\nPlotting {n_samples} random samples...")

        for i in range(min(n_samples, len(pred_output))):
            idx = np.random.randint(0, len(pred_output))

            # Get encoder and prediction
            encoder_data = pred_x['encoder_cont'][idx, :, 0].cpu().numpy()  # Energy values
            actual = pred_y[0][idx].cpu().numpy()
            pred_median = pred_output[idx, :, 3].cpu().numpy()  # 50th percentile
            pred_10 = pred_output[idx, :, 0].cpu().numpy()      # 10th percentile
            pred_90 = pred_output[idx, :, 6].cpu().numpy()      # 90th percentile

            # Plot
            fig, ax = plt.subplots(figsize=(12, 5))

            # Historical (encoder)
            encoder_x = np.arange(-len(encoder_data), 0)
            ax.plot(encoder_x, encoder_data, 'b-', label='Historical', linewidth=2)

            # Actual future
            pred_x = np.arange(0, len(actual))
            ax.plot(pred_x, actual, 'g-', label='Actual', linewidth=2, marker='o')

            # Predicted future
            ax.plot(pred_x, pred_median, 'r--', label='Predicted (median)', linewidth=2)
            ax.fill_between(pred_x, pred_10, pred_90, alpha=0.3, color='red',
                            label='10-90% prediction interval')

            ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Energy Consumption (MW)', fontsize=12)
            ax.set_title(f'Sample {idx}: 24-Hour Ahead Forecast', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = f'prediction_sample_{i+1}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: {output_file}")
            plt.close()

    # Calculate metrics
    actuals_flat = pred_y[0].cpu().numpy().flatten()
    preds_flat = pred_output[:, :, 3].cpu().numpy().flatten()  # Median

    mae = np.mean(np.abs(actuals_flat - preds_flat))
    rmse = np.sqrt(np.mean((actuals_flat - preds_flat) ** 2))
    mape = np.mean(np.abs((actuals_flat - preds_flat) / actuals_flat)) * 100

    print("\n" + "=" * 60)
    print("Test Set Metrics:")
    print("-" * 60)
    print(f"  MAE:  {mae:>10.2f} MW")
    print(f"  RMSE: {rmse:>10.2f} MW")
    print(f"  MAPE: {mape:>10.2f} %")
    print("=" * 60)

    return predictions


def predict_future(checkpoint_path, n_days=7):
    """Make true future predictions (beyond available data)."""

    print(f"\nMaking {n_days}-day future forecast...")

    # Load model
    best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

    # Load most recent data
    df = load_and_prepare_data()

    # Take last 168 hours as encoder context
    encoder_length = 168
    prediction_length = 24

    recent_data = df.tail(encoder_length + prediction_length).copy()

    print(f"Using data from {recent_data['Datetime'].min()} to {recent_data['Datetime'].max()}")

    # For true future prediction, you'd extend the dataframe with future timesteps
    # and provide known features (hour, month, etc.) but leave unknowns as NaN

    # This is a simplified example - full implementation would require
    # iterative prediction for multi-day forecasts

    print("Note: Full future prediction requires iterative forecasting.")
    print("See exploration.ipynb for complete implementation.")

    return None


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained TFT')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of sample plots to generate')
    parser.add_argument('--future', action='store_true',
                        help='Make future predictions (not just test set)')

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ ERROR: Checkpoint not found: {args.checkpoint}")
        return

    if args.future:
        predict_future(args.checkpoint)
    else:
        make_predictions(args.checkpoint, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
