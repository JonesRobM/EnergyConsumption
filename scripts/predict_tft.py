"""
Simple script to make predictions using a trained TFT model.

Usage:
    python predict_tft.py --checkpoint checkpoints/tft-epoch=15-val_loss=0.45.ckpt
"""

import argparse
import pandas as pd
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch # Added missing import

from pytorch_forecasting import TemporalFusionTransformer
from train_tft import load_and_prepare_data, create_datasets, create_model


def make_predictions(checkpoint_path, training_dataset, test_dataset, n_samples=5, save_plots=True):
    """Load model and make predictions on test set."""

    print(f"Loading model from: {checkpoint_path}")

    # Manually load checkpoint with weights_only=False
    # This bypasses the problematic pytorch_lightning load_from_checkpoint for compatibility
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract hyperparameters from checkpoint
    hparams = checkpoint["hyper_parameters"]

    # Re-initialize the model with the same architecture parameters
    best_tft = create_model(
        training_dataset, # Use the passed argument
        hidden_size=hparams["hidden_size"],
        attention_heads=hparams["attention_head_size"], # Use 'attention_head_size' from hparams
        dropout=hparams["dropout"],
        learning_rate=hparams["learning_rate"]
    )

    # Load the state_dict into the initialized model
    best_tft.load_state_dict(checkpoint["state_dict"])
    best_tft.eval() # Set to evaluation mode

    # Create test dataloader
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    # Make predictions
    print(f"Making predictions on {len(test_dataset)} test samples...")
    # Use mode="raw" to get detailed outputs for plotting and interpretation
    predictions = best_tft.predict(
        test_dataloader,
        mode="raw",
        return_x=True,
        return_y=True,
    )

    # Extract results
    raw_predictions = predictions.output  # This is a dictionary of raw outputs (prediction, attention, etc.)
    pred_output_quantiles = raw_predictions['prediction'] # The actual predicted quantiles
    pred_x = predictions.x  # x_data used for predictions
    pred_y = predictions.y  # Actual values

    print(f"\nPrediction shape (quantiles): {pred_output_quantiles.shape}")
    print(f"  - {pred_output_quantiles.shape[0]} samples")
    print(f"  - {pred_output_quantiles.shape[1]} timesteps ahead")
    print(f"  - {pred_output_quantiles.shape[2]} quantiles")


    if save_plots:
        # Plotting variable importances
        print("\nPlotting variable importances...")
        try:
            # We need to pick a single sample's x data and raw_predictions for interpret_output
            # Let's get a sample from the first batch
            first_batch_x, _ = next(iter(test_dataloader))
            first_batch_raw_predictions = best_tft.predict(first_batch_x, mode="raw")

            # Get interpretation for the first sample in the first batch
            sample_idx_for_interpretation = 0
            single_sample_raw_predictions = {
                key: value[sample_idx_for_interpretation].unsqueeze(0) if isinstance(value, torch.Tensor) else value
                for key, value in first_batch_raw_predictions.items()
            }
            single_sample_x = {
                key: value[sample_idx_for_interpretation].unsqueeze(0) if isinstance(value, torch.Tensor) else value
                for key, value in first_batch_x.items()
            }

            interpretation = best_tft.interpret_output(single_sample_raw_predictions, single_sample_x)
            fig_importance = best_tft.plot_interpretation(interpretation)
            fig_importance.savefig('variable_importances.png', dpi=150, bbox_inches='tight')
            print("  ✓ Saved: variable_importances.png")
            plt.close(fig_importance)
        except Exception as e:
            print(f"  ✗ Could not plot variable importances: {e}")


        # Plot random samples using built-in method
        print(f"\nPlotting {n_samples} random samples...")
        
        
        # Determine the index of the target variable in the 'encoder_cont' tensor
        target_name = training_dataset.target_names[0]
        target_idx_in_reals = training_dataset.time_varying_unknown_reals.index(target_name)

        for i in range(min(n_samples, pred_output_quantiles.shape[0])): # Use shape[0] for actual samples
            idx = np.random.randint(0, pred_output_quantiles.shape[0]) # Random index within the batch

            # Get encoder and prediction
            # encoder_data from pred_x['encoder_cont'] needs to be extracted correctly for 'idx'
            # pred_x['encoder_cont'] is shape [batch_size, encoder_length, num_encoder_reals]
            encoder_data = pred_x['encoder_cont'][idx, :, target_idx_in_reals].cpu().numpy() # Extract target from encoder

            actual = pred_y[0][idx].cpu().numpy() # Actuals for prediction horizon

            pred_median = pred_output_quantiles[idx, :, pred_output_quantiles.shape[2] // 2].cpu().numpy()  # Median
            pred_10 = pred_output_quantiles[idx, :, 0].cpu().numpy()      # 10th percentile
            pred_90 = pred_output_quantiles[idx, :, -1].cpu().numpy()      # 90th percentile (assuming sorted quantiles, last one)

            # Plot
            fig, ax = plt.subplots(figsize=(12, 5))

            # Historical (encoder)
            encoder_x = np.arange(-len(encoder_data), 0)
            ax.plot(encoder_x, encoder_data, 'b-', label='Historical', linewidth=2)

            # Actual future
            pred_x_axis = np.arange(0, len(actual))
            ax.plot(pred_x_axis, actual, 'g-', label='Actual', linewidth=2, marker='o')

            # Predicted future
            ax.plot(pred_x_axis, pred_median, 'r--', label='Predicted (median)', linewidth=2)
            ax.fill_between(pred_x_axis, pred_10, pred_90, alpha=0.3, color='red',
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
    # pred_output_quantiles contains quantiles, so we need to pick the median (e.g., 50th percentile)
    # The quantiles are usually sorted, so the median is at index n_quantiles // 2
    median_idx = pred_output_quantiles.shape[2] // 2
    preds_flat = pred_output_quantiles[:, :, median_idx].cpu().numpy().flatten()
    actuals_flat = pred_y[0].cpu().numpy().flatten() # pred_y[0] should be the actuals tensor

    # Prepare DataFrame for detailed metric breakdown
    results_df = pd.DataFrame({
        'actual': actuals_flat,
        'prediction': preds_flat,
    })

    # Extract hour and dayofweek for each predicted step
    # pred_x['decoder_cat'] has shape [batch_size, prediction_length, num_decoder_categoricals]
    # We use the order defined in train_tft.py: time_varying_known_categoricals=["hour", "dayofweek", "month"]
    decoder_hours = pred_x['decoder_cat'][:, :, 0].cpu().numpy().flatten()
    decoder_dayofweek = pred_x['decoder_cat'][:, :, 1].cpu().numpy().flatten()

    results_df['hour'] = decoder_hours
    results_df['dayofweek'] = decoder_dayofweek

    # Add prediction horizon (0-indexed, 0 = 1st hour ahead, 23 = 24th hour ahead)
    horizon_idx = np.tile(np.arange(pred_output_quantiles.shape[1]), pred_output_quantiles.shape[0])
    results_df['horizon'] = horizon_idx

    # Calculate overall metrics
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

    # Calculate Prediction Interval Coverage Error (PICE)
    if pred_output_quantiles.shape[2] > 2: # Check if quantiles are available (at least 3 for 10, 50, 90)
        # Assuming quantiles are sorted: 10th percentile is index 0, 90th percentile is index -1 (or 6 for 7 quantiles)
        q_10 = pred_output_quantiles[:, :, 0].cpu().numpy().flatten()
        q_90 = pred_output_quantiles[:, :, -1].cpu().numpy().flatten() # Assuming 7 quantiles, -1 is 90th

        actuals_flat = pred_y[0].cpu().numpy().flatten()

        # Count how many actuals fall within the 10-90 interval
        in_interval = ((actuals_flat >= q_10) & (actuals_flat <= q_90)).sum()
        total_predictions = actuals_flat.shape[0]
        pice = (in_interval / total_predictions) * 100

        print(f"  PICE (10-90%): {pice:>10.2f} % (Expected: 80.00 %)")
    else:
        print("  PICE: Not enough quantiles to calculate 10-90% interval.")

    print("=" * 60) # Re-add separator


    print("\nDetailed Metric Breakdowns:")
    print("-" * 60)

    # Function to calculate metrics for a group
    def calculate_group_metrics(group_df):
        abs_error = np.abs(group_df['actual'] - group_df['prediction'])
        sq_error = (group_df['actual'] - group_df['prediction']) ** 2
        
        group_mae = np.mean(abs_error)
        group_rmse = np.sqrt(np.mean(sq_error))
        group_mape = np.mean(abs_error / group_df['actual']) * 100
        return pd.Series({'MAE': group_mae, 'RMSE': group_rmse, 'MAPE': group_mape})

    # Metrics by Prediction Horizon
    print("\nMetrics by Prediction Horizon (hours ahead):")
    horizon_metrics = results_df.groupby('horizon').apply(calculate_group_metrics)
    print(horizon_metrics.to_string())

    # Metrics by Hour of Day
    print("\nMetrics by Hour of Day:")
    hour_metrics = results_df.groupby('hour').apply(calculate_group_metrics)
    print(hour_metrics.to_string())

    # Metrics by Day of Week
    print("\nMetrics by Day of Week (0=Monday, 6=Sunday):")
    dayofweek_metrics = results_df.groupby('dayofweek').apply(calculate_group_metrics)
    print(dayofweek_metrics.to_string())
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

    # Load data and create datasets
    print("Loading data...")
    df = load_and_prepare_data()
    training, validation, test = create_datasets(df)

    if args.future:
        predict_future(args.checkpoint)
    else:
        make_predictions(args.checkpoint, training, test, n_samples=args.n_samples) # training is now defined here


if __name__ == "__main__":
    main()
