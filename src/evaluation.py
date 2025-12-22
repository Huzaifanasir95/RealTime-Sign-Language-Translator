"""
Evaluation and metrics utilities.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from pathlib import Path
from typing import List, Tuple, Optional
import json


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     class_names: List[str]) -> dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
            per_class_metrics[class_name] = {
                'accuracy': float(class_acc),
                'support': int(class_mask.sum())
            }
    
    metrics = {
        'overall_accuracy': float(accuracy),
        'weighted_precision': float(precision),
        'weighted_recall': float(recall),
        'weighted_f1': float(f1),
        'per_class_metrics': per_class_metrics
    }
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str],
                          output_path: Path,
                          normalize: bool = False,
                          figsize: Tuple[int, int] = (20, 18)):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save plot
        normalize: Whether to normalize the matrix
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_accuracy(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            class_names: List[str],
                            output_path: Path,
                            figsize: Tuple[int, int] = (16, 6)):
    """
    Plot per-class accuracy bar chart.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save plot
        figsize: Figure size
    """
    accuracies = []
    for i in range(len(class_names)):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
            accuracies.append(class_acc)
        else:
            accuracies.append(0)
    
    # Color code by accuracy
    colors = ['green' if acc >= 0.9 else 'orange' if acc >= 0.7 else 'red' 
              for acc in accuracies]
    
    plt.figure(figsize=figsize)
    plt.bar(class_names, accuracies, color=colors, edgecolor='black', alpha=0.7)
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1.05])
    plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history_dict: dict,
                          output_path: Path,
                          figsize: Tuple[int, int] = (14, 5)):
    """
    Plot training history (accuracy and loss).
    
    Args:
        history_dict: Training history dictionary
        output_path: Path to save plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy plot
    if 'accuracy' in history_dict:
        axes[0].plot(history_dict['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(history_dict['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    if 'loss' in history_dict:
        axes[1].plot(history_dict['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(history_dict['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('Loss', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_classification_report(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               class_names: List[str],
                               output_path: Path):
    """
    Save detailed classification report to file.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save report
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names, 
                                   digits=4)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(report)
        f.write("\n\n")
        f.write("="*70 + "\n")
        f.write("CONFUSION MATRIX STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        cm = confusion_matrix(y_true, y_pred)
        f.write(f"Total samples: {len(y_true)}\n")
        f.write(f"Correct predictions: {(y_true == y_pred).sum()}\n")
        f.write(f"Incorrect predictions: {(y_true != y_pred).sum()}\n")
        f.write(f"Overall accuracy: {accuracy_score(y_true, y_pred):.4f}\n")


def save_metrics_json(metrics: dict, output_path: Path):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_path: Path to save JSON
    """
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def find_misclassified_samples(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               predictions: np.ndarray,
                               top_n: int = 10) -> List[dict]:
    """
    Find top misclassified samples.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        predictions: Prediction probabilities
        top_n: Number of top samples to return
        
    Returns:
        List of misclassified sample info
    """
    # Find misclassified indices
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    # Get confidence for misclassified samples
    misclassified_info = []
    for idx in misclassified_idx:
        confidence = predictions[idx][y_pred[idx]]
        misclassified_info.append({
            'index': int(idx),
            'true_label': int(y_true[idx]),
            'predicted_label': int(y_pred[idx]),
            'confidence': float(confidence)
        })
    
    # Sort by confidence (highest confidence mistakes first)
    misclassified_info.sort(key=lambda x: x['confidence'], reverse=True)
    
    return misclassified_info[:top_n]


if __name__ == "__main__":
    print("Evaluation utilities loaded successfully!")
