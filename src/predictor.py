import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
import joblib

from src.logger import setup_logger
from src.model_trainer import ModelTrainer
from src.feature_engineer import prepare_features

logger = setup_logger(__name__, "predictor.log")


class FraudPredictor:
    """Handles predictions for fraud detection."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model (loaded if provided)
        """
        self.model_trainer = None
        self.dataset_type = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Path) -> None:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model
        """
        logger.info(f"Loading model from {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Infer dataset type from filename
        filename = model_path.stem
        if 'fraud' in filename.lower():
            self.dataset_type = 'fraud'
        elif 'credit' in filename.lower():
            self.dataset_type = 'creditcard'
        else:
            logger.warning("Could not infer dataset type from filename")
        
        # Load model
        self.model_trainer = ModelTrainer()
        self.model_trainer.load_model(model_path)
        
        logger.info(f"Model loaded: {self.model_trainer.model_type}")
    
    def predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = False,
        threshold: float = 0.5
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
            return_proba: Whether to return probabilities
            threshold: Classification threshold
        
        Returns:
            Predictions or dict with predictions and probabilities
        """
        if self.model_trainer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        logger.info(f"Making predictions on {len(X)} samples")
        
        # Validate features
        self._validate_features(X)
        
        # Get predictions
        y_pred_proba = self.model_trainer.predict_proba(X)
        y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
        
        logger.info(
            f"Predictions complete: "
            f"{y_pred.sum()} fraud cases detected ({y_pred.mean()*100:.2f}%)"
        )
        
        if return_proba:
            return {
                'predictions': y_pred,
                'probabilities': y_pred_proba[:, 1],
                'fraud_probability': y_pred_proba[:, 1]
            }
        
        return y_pred
    
    def predict_single(
        self,
        transaction: Dict[str, Any],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predict fraud for a single transaction.
        
        Args:
            transaction: Dictionary with transaction features
            threshold: Classification threshold
        
        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        X = pd.DataFrame([transaction])
        
        # Make prediction
        result = self.predict(X, return_proba=True, threshold=threshold)
        
        prediction_result = {
            'is_fraud': bool(result['predictions'][0]),
            'fraud_probability': float(result['fraud_probability'][0]),
            'confidence': float(
                max(result['probabilities'][0], 1 - result['probabilities'][0])
            ),
            'threshold': threshold
        }
        
        logger.info(
            f"Single prediction: fraud={prediction_result['is_fraud']}, "
            f"probability={prediction_result['fraud_probability']:.4f}"
        )
        
        return prediction_result
    
    def find_optimal_threshold(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        metric: str = 'f1',
        min_recall: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Find optimal classification threshold for fraud detection.
        
        Args:
            X: Features
            y_true: True labels
            metric: Metric to optimize ('f1', 'recall', 'precision')
            min_recall: Minimum recall constraint (e.g., 0.8 for at least 80% recall)
        
        Returns:
            Dictionary with optimal threshold and performance metrics
        """
        from sklearn.metrics import precision_recall_curve
        
        if self.model_trainer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        logger.info(f"Finding optimal threshold based on {metric}")
        
        # Get prediction probabilities
        y_pred_proba = self.model_trainer.predict_proba(X)
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(
            y_true, y_pred_proba[:, 1]
        )
        
        # Calculate F1 scores
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Find optimal threshold based on metric with constraints
        if metric == 'f1':
            if min_recall is not None:
                # Find best F1 while maintaining minimum recall
                valid_indices = recalls >= min_recall
                if valid_indices.any():
                    valid_f1 = np.where(valid_indices, f1_scores, -np.inf)
                    optimal_idx = np.argmax(valid_f1)
                    logger.info(f"Optimizing F1 with min_recall >= {min_recall}")
                else:
                    logger.warning(f"No threshold achieves min_recall={min_recall}, using best F1")
                    optimal_idx = np.argmax(f1_scores)
            else:
                optimal_idx = np.argmax(f1_scores)
        elif metric == 'recall':
            optimal_idx = np.argmax(recalls)
        elif metric == 'precision':
            optimal_idx = np.argmax(precisions)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Get threshold (handling edge case)
        if optimal_idx < len(thresholds):
            optimal_threshold = float(thresholds[optimal_idx])
        else:
            optimal_threshold = float(thresholds[-1])
        
        # Evaluate at optimal threshold
        y_pred_optimal = (y_pred_proba[:, 1] >= optimal_threshold).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        result = {
            'optimal_threshold': optimal_threshold,
            'accuracy': float(accuracy_score(y_true, y_pred_optimal)),
            'precision': float(precision_score(y_true, y_pred_optimal, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_optimal, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred_optimal, zero_division=0)),
            'optimization_metric': metric
        }
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"Metrics at optimal threshold:")
        logger.info(f"  Precision: {result['precision']:.4f}")
        logger.info(f"  Recall: {result['recall']:.4f}")
        logger.info(f"  F1 Score: {result['f1_score']:.4f}")
        
        return result
    
    def predict_batch(
        self,
        transactions: pd.DataFrame,
        threshold: float = 0.5,
        include_details: bool = False
    ) -> pd.DataFrame:
        """
        Predict fraud for a batch of transactions.
        
        Args:
            transactions: DataFrame with transaction features
            threshold: Classification threshold
            include_details: Whether to include detailed probabilities
        
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Processing batch of {len(transactions)} transactions")
        
        results = self.predict(transactions, return_proba=True, threshold=threshold)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'is_fraud': results['predictions'],
            'fraud_probability': results['fraud_probability']
        })
        
        if include_details:
            results_df['confidence'] = results_df['fraud_probability'].apply(
                lambda x: max(x, 1 - x)
            )
            results_df['risk_level'] = results_df['fraud_probability'].apply(
                self._get_risk_level
            )
        
        return results_df
    
    def _validate_features(self, X: pd.DataFrame) -> None:
        """
        Validate that input features match trained model.
        
        Args:
            X: Input features
        
        Raises:
            ValueError: If features don't match
        """
        expected_features = set(self.model_trainer.feature_names)
        actual_features = set(X.columns)
        
        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                X[feature] = 0
        
        if extra_features:
            logger.warning(f"Extra features will be ignored: {extra_features}")
            X = X[self.model_trainer.feature_names]
        
        # Ensure correct order
        X = X[self.model_trainer.feature_names]
    
    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """
        Convert probability to risk level.
        
        Args:
            probability: Fraud probability
        
        Returns:
            Risk level string
        """
        if probability < 0.3:
            return 'low'
        elif probability < 0.6:
            return 'medium'
        elif probability < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get feature contributions for predictions.
        
        Args:
            X: Features to explain
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature contributions
        """
        if self.model_trainer is None:
            raise RuntimeError("No model loaded")
        
        # Get feature importance
        feature_importance = self.model_trainer.get_feature_importance()
        
        if feature_importance is None:
            logger.warning("Feature importance not available for this model")
            return pd.DataFrame()
        
        # Get top features
        top_features = feature_importance.head(top_n)
        
        logger.info(f"Top {top_n} features for fraud detection:")
        for _, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return top_features


def load_predictor(
    model_type: str,
    dataset_type: str,
    models_dir: Optional[Path] = None
) -> FraudPredictor:
    """
    Convenience function to load a predictor.
    
    Supports both naming conventions:
    - Notebook convention: {dataset_prefix}_{model_type}_model.joblib (e.g., ecommerce_xgboost_model.joblib)
    - Dashboard convention: {model_type}_{dataset_type}_model.joblib (e.g., xgboost_fraud_model.joblib)
    
    Args:
        model_type: Type of model ('logistic_regression', 'random_forest', 'xgboost')
        dataset_type: Type of dataset ('fraud', 'creditcard')
        models_dir: Directory containing models
    
    Returns:
        Loaded FraudPredictor
    """
    from src.config import model_config
    
    if models_dir is None:
        models_dir = model_config.model_save_dir
    
    # Map dataset_type to notebook prefix
    prefix_map = {'fraud': 'ecommerce', 'creditcard': 'creditcard'}
    notebook_prefix = prefix_map.get(dataset_type, dataset_type)
    
    # Try notebook naming convention first (from trained notebooks)
    notebook_path = models_dir / f"{notebook_prefix}_{model_type}_model.joblib"
    if notebook_path.exists():
        logger.info(f"Loading model using notebook convention: {notebook_path}")
        return FraudPredictor(notebook_path)
    
    # Try dashboard naming convention
    dashboard_path = models_dir / f"{model_type}_{dataset_type}_model.joblib"
    if dashboard_path.exists():
        logger.info(f"Loading model using dashboard convention: {dashboard_path}")
        return FraudPredictor(dashboard_path)
    
    # Raise error with helpful message
    raise FileNotFoundError(
        f"Model not found. Searched for:\n"
        f"  - {notebook_path}\n"
        f"  - {dashboard_path}\n"
        f"Please train a model first using the notebooks or dashboard."
    )
