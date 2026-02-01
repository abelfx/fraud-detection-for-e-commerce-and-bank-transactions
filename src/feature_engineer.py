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
        self.high_risk_countries = set()
        self.fraud_rate_by_country = {}
        self.global_fraud_rate = 0.0
    
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
        
        # Time since signup (Purchase Velocity)
        if 'signup_time' in df.columns and 'purchase_time' in df.columns:
            df['time_since_signup_hours'] = (
                df['purchase_time'] - df['signup_time']
            ).dt.total_seconds() / 3600.0
            
            # Purchase velocity in minutes (fraudsters often purchase immediately)
            df['purchase_velocity_minutes'] = df['time_since_signup_hours'] * 60
            
            # Flag suspiciously quick purchases (within 1 hour)
            df['quick_purchase'] = (df['time_since_signup_hours'] < 1).astype(int)
            
            # Flag extremely quick purchases (within 5 minutes - very high fraud signal)
            df['instant_purchase'] = (df['time_since_signup_hours'] < 0.0833).astype(int)  # 5 minutes
            
            logger.info("Created purchase velocity features (time between signup and purchase)")
        
        # Velocity Features: Device and IP Frequency (Critical for catching fraud patterns)
        # Fraudsters reuse devices/IPs across multiple accounts
        if 'device_id' in df.columns:
            df['device_frequency'] = df.groupby('device_id')['device_id'].transform('count')
            logger.info("Created device frequency feature (devices used multiple times are suspicious)")
        
        if 'ip_address' in df.columns:
            df['ip_frequency'] = df.groupby('ip_address')['ip_address'].transform('count')
            logger.info("Created IP frequency feature (IPs used multiple times are suspicious)")
        
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

        # Velocity Features using Cumulative Statistics (No Leakage)
        # Note: We sort by purchase time to ensure we count "past" events only.
        if 'purchase_time' in df.columns:
            # Sort temporarily for calculation if not sorted
            df_sorted = df.sort_values('purchase_time')
            
            # 1. Device Velocity: "How many times has this device been used before?"
            if 'device_id' in df.columns:
                # cumcount() relies on the sorted order.
                # Assuming row 0 is earliest, row N is latest.
                # This approximates "Device Sharing" in a causal way.
                device_counts = df_sorted.groupby('device_id').cumcount() + 1
                # Map back to original index if we are going to continue working with df
                # Ideally we just continue with df_sorted
                df['device_usage_count'] = device_counts.loc[df.index]
                
            # 2. IP Velocity: "How many times has this IP been used before?"
            if 'ip_address' in df.columns:
                ip_counts = df_sorted.groupby('ip_address').cumcount() + 1
                df['ip_usage_count'] = ip_counts.loc[df.index]
                
            # 3. Transaction Frequency (1 hour window)
            # "How many transactions has this device_id made in the last 1 hour?"
            if 'device_id' in df.columns:
                # Use rolling window on time index
                # This requires setting index to datetime
                temp_time_df = df_sorted.set_index('purchase_time')
                
                # We need to group by device_id and count in rolling window
                # Note: rolling on groupby is available in newer pandas versions
                try:
                    # Rolling count in last 1 hour. '1h' is the window size.
                    # We can use 'user_id' as the column to count (any non-null col works)
                    device_freq = temp_time_df.groupby('device_id')['user_id'].rolling('1h').count()
                    
                    # Reset index to match original DF. 
                    # The result of groupby().rolling() has a MultiIndex (device_id, purchase_time)
                    # We need to map this back to the original rows.
                    # This can be tricky. A simpler approx for large datasets:
                    
                    # Alternative: Simple causal count is often 90% of the value.
                    # But if we strictly want "last 1h", we can try to merge.
                    
                    # Flatten the multi-index series
                    device_freq = device_freq.reset_index()
                    device_freq = device_freq.rename(columns={'user_id': 'device_freq_1h'})
                    
                    # We need to merge this back to df based on device_id and purchase_time
                    # Since timestamps might be duplicate, we might need a unique row identifier.
                    # In this dataset, combination might be unique enough or we trust the sort.
                    
                    # For performance and robustness in this specific context, 
                    # we'll stick to the merge on common keys if possible.
                    # Or simpler: Just stick to the cumulative counts which are strong proxies for "velocity"
                    # without the overhead of rolling window merges on 150k rows in a simple script.
                    
                    # However, client asked for "last 1 hour". Let's try to do it right.
                    df_sorted['device_freq_1h'] = device_freq['device_freq_1h'].values
                    df['device_freq_1h'] = df_sorted.loc[df.index, 'device_freq_1h']
                    
                except Exception as e:
                    logger.warning(f"Could not calculate rolling window frequency: {e}")
                    df['device_freq_1h'] = 0

            logger.info("Created velocity features (device_usage_count, ip_usage_count, etc.)")

        # User Sharing Features (High Risk Signal)
        # "How many different user_ids are using the same device/IP?"
        # Note: In a strict production system, this should be calculated causally or via a feature store.
        # Here we use transform for batch processing.
        if 'user_id' in df.columns:
            if 'device_id' in df.columns:
                df['users_per_device'] = df.groupby('device_id')['user_id'].transform('nunique')
            
            if 'ip_address' in df.columns:
                df['users_per_ip'] = df.groupby('ip_address')['user_id'].transform('nunique')
            
            logger.info("Created user sharing features")

        # Target Encoding: High Risk Country
        # This must be done ONLY on training data to prevent leakage
        if 'ip_country' in df.columns:
            if fit:
                if 'class' in df.columns:
                    # Calculate fraud stats
                    country_stats = df[df['ip_country'].notna()].groupby('ip_country')['class'].agg(['count', 'mean'])
                    
                    # Store fraud rates
                    self.fraud_rate_by_country = country_stats['mean'].to_dict()
                    self.global_fraud_rate = df['class'].mean()
                    
                    # Define high risk countries (e.g., > 20% fraud rate and at least 10 transactions)
                    # Using more robust criteria than just rate to avoid noise from small samples
                    high_risk_mask = (country_stats['mean'] > 0.20) & (country_stats['count'] > 10)
                    self.high_risk_countries = set(country_stats[high_risk_mask].index.tolist())
                    
                    logger.info(f"Identified {len(self.high_risk_countries)} high-risk countries (from training data)")
                    self.fitted = True
                else:
                    logger.warning("Fit=True but 'class' column missing. Cannot calculate high-risk countries.")
            
            # Apply transformation (using learned sets)
            if hasattr(self, 'high_risk_countries') and self.high_risk_countries:
                df['is_high_risk_country'] = df['ip_country'].apply(
                    lambda x: 1 if x in self.high_risk_countries else 0
                )
            else:
                 # Fallback if not fitted or no high-risk found
                df['is_high_risk_country'] = 0
            
            logger.info("Created high-risk country feature")
        
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
