import os
import glob
import pandas as pd
import kagglehub

COMPOSITE_DATA_PATH = 'composite_energy_data.csv'
DATA_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'kagglehub', 'datasets', 'robikscube', 'hourly-energy-consumption', 'versions', '3')

def download_and_prepare_data():
    """
    Downloads data from Kaggle and prepares the composite dataframe if it doesn't exist.
    If it exists, it loads it from the file.
    """
    if os.path.exists(COMPOSITE_DATA_PATH):
        print(f"'{COMPOSITE_DATA_PATH}' already exists. Loading from file.")
        df_composite = pd.read_csv(COMPOSITE_DATA_PATH)
        df_composite['Datetime'] = pd.to_datetime(df_composite['Datetime'])
        df_composite = df_composite.set_index('Datetime')
        return df_composite

    print("Downloading dataset from Kaggle...")
    try:
        kagglehub.dataset_download("robikscube/hourly-energy-consumption")
    except Exception as e:
        print(f"Could not download dataset from Kaggle. Please ensure your Kaggle API token is set up correctly. Error: {e}")
        return None

    print("Creating composite dataframe...")
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['source_file'] = os.path.basename(file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not dataframes:
        print("No CSV files found.")
        return None

    composite_df = pd.concat(dataframes, ignore_index=True)
    composite_df.to_csv(COMPOSITE_DATA_PATH, index=False)
    print(f"Composite dataframe saved as '{COMPOSITE_DATA_PATH}'")
    
    composite_df['Datetime'] = pd.to_datetime(composite_df['Datetime'])
    composite_df = composite_df.set_index('Datetime')
    return composite_df

if __name__ == '__main__':
    df = download_and_prepare_data()
    if df is not None:
        print("\nData loaded successfully.")
        print(df.info())
