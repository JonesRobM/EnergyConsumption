"""
Orchestrates training of Temporal Fusion Transformer models for multiple energy regions.

This script iterates through a predefined list of energy regions and calls the
`train_tft.py` script for each one, effectively training a separate model
per region.
"""

import subprocess
import sys
from pathlib import Path

# List of all energy regions to be trained
# These are the column names from `composite_energy_data.csv`
REGIONS = [
    "AEP_MW",
    "COMED_MW",
    "DAYTON_MW",
    "DEOK_MW",
    "DOM_MW",
    "DUQ_MW",
    "EKPC_MW",
    "FE_MW",
    "NI_MW",
    "PJME_MW",
    "PJMW_MW",
]

# Path to the standalone training script
TRAIN_SCRIPT_PATH = Path(__file__).parent / "train_tft.py"

def main():
    """
    Main function to loop through regions and launch training processes.
    """
    if not TRAIN_SCRIPT_PATH.exists():
        print(f"❌ Error: Training script not found at {TRAIN_SCRIPT_PATH}")
        sys.exit(1)

    print(f"🌍 Starting multi-region training for {len(REGIONS)} regions.")
    print("=" * 80)

    # You can customize the training parameters here
    # For a quick test, use --epochs 1
    # For a full run, you might use --epochs 50 or more
    common_args = [
        "--mode", "train_test",
        "--epochs", "1", # Using 1 epoch for demonstration
        "--batch_size", "128",
    ]

    for i, region in enumerate(REGIONS):
        print(f"\n▶️  ({i+1}/{len(REGIONS)}) Training model for region: {region}")
        print("-" * 80)

        # Construct the command for the training script
        command = [
            sys.executable,  # Use the same python interpreter
            str(TRAIN_SCRIPT_PATH),
            "--region", region,
        ] + common_args

        try:
            # Execute the training script as a subprocess
            subprocess.run(command, check=True)
            print(f"\n✅ Successfully completed training for {region}")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error training {region}: Subprocess returned error code {e.returncode}")
        except Exception as e:
            print(f"❌ An unexpected error occurred while training {region}: {e}")
        
        print("=" * 80)

    print("🏁 Multi-region training complete!")

if __name__ == "__main__":
    main()
