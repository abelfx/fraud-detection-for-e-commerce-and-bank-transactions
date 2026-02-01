import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import setup_logger

logger = setup_logger(__name__, "model_evaluator.log")


class ModelEvaluator:
    """Handles evaluation and scoring of fraud detection models."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def evaluate(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        y_pred: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model predictions.
        """
        logger.info("Evaluating model performance")
        
        # Generate predictions if not provided
        if y_pred is None:
            y_pred = (y_pred_proba[:, 1] >= self.threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba[:, 1]),
            'average_precision': average_precision_score(y_true, y_pred_proba[:, 1]),
            'threshold': self.threshold
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Calculate specificity (True Negative Rate)
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Log metrics
        logger.info("\n" + "="*60)
        logger.info("EVALUATION METRICS")
        logger.info("="*60)
        logger.info(f"Accuracy:           {metrics['accuracy']:.4f}")
        logger.info(f"Precision:          {metrics['precision']:.4f}")
        logger.info(f"Recall:             {metrics['recall']:.4f}")
        logger.info(f"F1 Score:           {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC:            {metrics['roc_auc']:.4f}")
        logger.info(f"AUPRC (AP):         {metrics['average_precision']:.4f} *** PRIMARY METRIC FOR IMBALANCED DATA ***")
        logger.info(f"Specificity:        {metrics['specificity']:.4f}")
        logger.info("="*60 + "\n")
        
        return metrics
    
    def find_optimal_threshold(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal classification threshold.
        """
        logger.info(f"Finding optimal threshold based on {metric}")
        
        # Get precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(
            y_true, y_pred_proba[:, 1]
        )
        
        # Calculate F1 scores
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Find optimal threshold based on metric
        if metric == 'f1':
            optimal_idx = np.argmax(f1_scores)
        elif metric == 'precision':
            optimal_idx = np.argmax(precisions)
        elif metric == 'recall':
            optimal_idx = np.argmax(recalls)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Get threshold (handling edge case)
        if optimal_idx < len(thresholds):
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = thresholds[-1]
        
        # Evaluate at optimal threshold
        y_pred_optimal = (y_pred_proba[:, 1] >= optimal_threshold).astype(int)
        
        metrics = {
            'threshold': optimal_threshold,
            'precision': precisions[optimal_idx],
            'recall': recalls[optimal_idx],
            'f1_score': f1_scores[optimal_idx]
        }
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"Metrics at optimal threshold: {metrics}")
        
        return optimal_threshold, metrics
    
    def generate_classification_report(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        target_names: Optional[list] = None
    ) -> str:
        """
        Generate detailed classification report.
        """
        if target_names is None:
            target_names = ['Normal', 'Fraud']
        
        report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            digits=4
        )
        
        logger.info(f"\nClassification Report:\n{report}")
        
        return report
    
    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = False
    ):
        """
        Plot confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_and_pr_curves(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot both ROC and Precision-Recall curves side by side.
        Emphasizes AUPRC as the primary metric for imbalanced data.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2, color='blue')
        ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax1.set_xlabel('False Positive Rate', fontsize=11)
        ax1.set_ylabel('True Positive Rate', fontsize=11)
        ax1.set_title('ROC Curve', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
        ap_score = average_precision_score(y_true, y_pred_proba[:, 1])
        baseline = y_true.sum() / len(y_true)
        
        ax2.plot(recall, precision, label=f'PR Curve (AUPRC = {ap_score:.4f})', linewidth=2, color='red')
        ax2.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline (Random) = {baseline:.4f}')
        ax2.set_xlabel('Recall', fontsize=11)
        ax2.set_ylabel('Precision', fontsize=11)
        ax2.set_title('Precision-Recall Curve\\n*** PRIMARY METRIC for Imbalanced Data ***', 
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC and PR curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot Precision-Recall curve.
        """
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_pred_proba[:, 1]
        )
        ap_score = average_precision_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.4f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """
        Create comparison plot for multiple models.
        """
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Prepare data for plotting
        model_names = list(results.keys())
        metrics_data = {
            metric: [results[model].get(metric, 0) for model in model_names]
            for metric in metrics_to_compare
        }
        
        # Create subplot for each metric
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_compare):
            ax = axes[idx]
            ax.bar(model_names, metrics_data[metric])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(metrics_data[metric]):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # Hide unused subplot
        axes[-1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def evaluate_model(
    model_trainer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimize_threshold: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a trained model.
    """
    evaluator = ModelEvaluator()
    
    # Get predictions
    y_pred_proba = model_trainer.predict_proba(X_test)
    y_pred = model_trainer.predict(X_test)
    
    # Evaluate
    metrics = evaluator.evaluate(y_test, y_pred_proba, y_pred)
    
    # Find optimal threshold if requested
    if optimize_threshold:
        optimal_threshold, optimal_metrics = evaluator.find_optimal_threshold(
            y_test, y_pred_proba, metric='f1'
        )
        metrics['optimal_threshold'] = optimal_threshold
        metrics['optimal_metrics'] = optimal_metrics
    
    # Generate classification report
    report = evaluator.generate_classification_report(y_test, y_pred)
    metrics['classification_report'] = report
    
    return metrics
