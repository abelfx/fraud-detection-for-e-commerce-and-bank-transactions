import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler

from src.logger import setup_logger

logger = setup_logger(__name__, "feature_engineer.log")


class FraudDataFeatureEngineer:
    """Feature engineering for fraud detection dataset."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.fitted = False
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        
        logger.info("Engineering features for fraud data")
        
        # Convert datetime columns
        if 'signup_time' in df.columns:
            df['signup_time'] = pd.to_datetime(df['signup_time'])
        if 'purchase_time' in df.columns:
            df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Time-based features
        if 'purchase_time' in df.columns:
            df['hour_of_day'] = df['purchase_time'].dt.hour
            df['day_of_week'] = df['purchase_time'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['day_of_month'] = df['purchase_time'].dt.day
            df['month'] = df['purchase_time'].dt.month
            
            logger.info("Created temporal features")
        
        # Time since signup
        if 'signup_time' in df.columns and 'purchase_time' in df.columns:
            df['time_since_signup_hours'] = (
                df['purchase_time'] - df['signup_time']
            ).dt.total_seconds() / 3600.0
            
            # Flag suspiciously quick purchases
            df['quick_purchase'] = (df['time_since_signup_hours'] < 1).astype(int)
            
            logger.info("Created time difference features")
        
        # IP address feature
        if 'ip_address' in df.columns:
            df['ip_as_int'] = df['ip_address'].apply(self._ip_to_int)
            logger.info("Converted IP addresses to integers")
        
        # Purchase value features
        if 'purchase_value' in df.columns:
            df['log_purchase_value'] = np.log1p(df['purchase_value'])
            df['purchase_value_rounded'] = (df['purchase_value'] % 1 == 0).astype(int)
            
            logger.info("Created purchase value features")
        
        # Age feature
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 18, 25, 35, 45, 55, 65, 100],
                labels=['<18', '18-25', '25-35', '35-45', '45-55', '55-65', '65+']
            )
            df['age_group'] = df['age_group'].cat.codes
            
            logger.info("Created age group features")
        
        # Browser and source interaction
        if 'browser' in df.columns and 'source' in df.columns:
            df['browser_source'] = df['browser'] + '_' + df['source']
            
            logger.info("Created interaction features")
        
        # One-hot encode categorical variables
        categorical_cols = ['browser', 'source', 'sex']
        existing_categorical = [col for col in categorical_cols if col in df.columns]
        
        if existing_categorical:
            df = pd.get_dummies(df, columns=existing_categorical, drop_first=True)
            logger.info(f"One-hot encoded: {existing_categorical}")
        
        return df
    
    @staticmethod
    def _ip_to_int(ip_str: str) -> int:
        """
        Convert IP address string to integer.
        """
        try:
            parts = ip_str.split('.')
            return (int(parts[0]) << 24) + (int(parts[1]) << 16) + \
                   (int(parts[2]) << 8) + int(parts[3])
        except:
            return 0


class CreditCardFeatureEngineer:
    """Feature engineering for credit card dataset."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.amount_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        self.fitted = False
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        
        logger.info("Engineering features for credit card data")
        
        # Scale Time feature
        if 'Time' in df.columns:
            if fit:
                df['scaled_time'] = self.time_scaler.fit_transform(
                    df['Time'].values.reshape(-1, 1)
                )
            else:
                df['scaled_time'] = self.time_scaler.transform(
                    df['Time'].values.reshape(-1, 1)
                )
            
            # Create time-based features
            df['hour'] = (df['Time'] / 3600) % 24
            df['day'] = (df['Time'] / 86400).astype(int)
            
            logger.info("Created and scaled time features")
        
        # Scale Amount feature
        if 'Amount' in df.columns:
            if fit:
                df['scaled_amount'] = self.amount_scaler.fit_transform(
                    df['Amount'].values.reshape(-1, 1)
                )
                self.fitted = True
            else:
                df['scaled_amount'] = self.amount_scaler.transform(
                    df['Amount'].values.reshape(-1, 1)
                )
            
            # Amount-based features
            df['log_amount'] = np.log1p(df['Amount'])
            df['amount_category'] = pd.cut(
                df['Amount'],
                bins=[-np.inf, 10, 50, 100, 500, np.inf],
                labels=['very_small', 'small', 'medium', 'large', 'very_large']
            )
            df['amount_category'] = df['amount_category'].cat.codes
            
            logger.info("Created and scaled amount features")
        
        # Interaction features with V columns
        v_columns = [col for col in df.columns if col.startswith('V')]
        if v_columns and 'scaled_amount' in df.columns:
            # Create a few key interactions
            for v_col in v_columns[:5]:  # Top 5 V features
                df[f'{v_col}_amount_interaction'] = df[v_col] * df['scaled_amount']
            
            logger.info("Created interaction features")
        
        return df


def prepare_features(
    df: pd.DataFrame,
    dataset_type: str,
    drop_columns: Optional[List[str]] = None,
    fit: bool = True
) -> pd.DataFrame:
    """
    Apply feature engineering based on dataset type.
    """
    if dataset_type == "fraud":
        engineer = FraudDataFeatureEngineer()
    elif dataset_type == "creditcard":
        engineer = CreditCardFeatureEngineer()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    df = engineer.engineer_features(df, fit=fit)
    
    # Drop specified columns
    if drop_columns:
        cols_to_drop = [col for col in drop_columns if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns: {cols_to_drop}")
    
    # Select only numeric columns for modeling
    numeric_df = df.select_dtypes(include=['number', 'bool'])
    logger.info(f"Final feature set: {len(numeric_df.columns)} columns")
    
    return numeric_df
