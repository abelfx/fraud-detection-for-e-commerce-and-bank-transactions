import pandas as pd
import joblib
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.config import data_config, model_config

def save_scaler():
    print("Loading local data to create the 'Translation Dictionary' (Scaler)...")
    
    # 1. Load Data
    loader = DataLoader()
    try:
        # We try to load processed data first as it's faster
        if data_config.fraud_processed_path.exists():
            print(f"Loading processed data from {data_config.fraud_processed_path}...")
            df = pd.read_csv(data_config.fraud_processed_path)
            dataset_name = 'fraud'
        else:
            print("Processed data not found. Attempting to load raw data...")
            # This follows the dashboard logic for feature engineering
            # Simplified for scaler fitting: we need the columns that the preprocessor expects
            # If processed data is missing locally, we might need to run the full pipeline.
            # Assuming you have the processed CSV since you ran it locally.
            print("Error: Please ensure 'data/processed/fraud_data_processed.csv' exists.")
            return
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Fit Scaler
    print("Fitting Scaler (calculating averages and deviations)...")
    preprocessor = DataPreprocessor(use_smote=False)
    
    # Prepare X (features)
    if data_config.target_column in df.columns:
        X = df.drop(columns=[data_config.target_column])
        y = df[data_config.target_column]
    else:
        X = df
        y = None # Not needed for scaling
        
    preprocessor.fit_transform(X, y)
    
    # 3. Save Scaler
    scaler_path = model_config.model_save_dir / "preprocessor_scaler.joblib"
    col_names_path = model_config.model_save_dir / "feature_columns.joblib"
    
    print(f"Saving Scaler to {scaler_path}...")
    joblib.dump(preprocessor, scaler_path)
    
    # Also save the column names so we know the exact order
    print(f"Saving feature column names to {col_names_path}...")
    joblib.dump(list(X.columns), col_names_path)
    
    print("\nSUCCESS! ✅")
    print("Now you can push these two new files to GitHub:")
    print(f"1. models/preprocessor_scaler.joblib")
    print(f"2. models/feature_columns.joblib")

if __name__ == "__main__":
    save_scaler()
