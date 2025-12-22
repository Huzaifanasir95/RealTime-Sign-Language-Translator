"""
Step 4: Evaluate Trained Model

This script evaluates the trained model and generates performance metrics.

Usage:
    conda activate timegan-gpu
    python scripts/step4_evaluate_model.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_model(model_path):
    """Load trained model."""
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70 + "\n")
    
    if not model_path.exists():
        print(f"[X] Model not found at: {model_path}")
        print("\nPlease train the model first:")
        print("  python scripts/step3_train_model.py")
        return None
    
    model = keras.models.load_model(model_path)
    print(f"[OK] Model loaded from: {model_path}")
    
    return model


def create_test_generator(data_dir, img_size=(224, 224), batch_size=32):
    """Create test data generator."""
    print("\n" + "="*70)
    print("CREATING TEST DATA GENERATOR")
    print("="*70 + "\n")
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"[OK] Test samples: {test_generator.samples}")
    print(f"[OK] Number of classes: {test_generator.num_classes}")
    
    return test_generator


def evaluate_model(model, test_gen):
    """Evaluate model on test data."""
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70 + "\n")
    
    print("[>>] Running evaluation...")
    results = model.evaluate(test_gen, verbose=1)
    
    print(f"\n[OK] Test Loss: {results[0]:.4f}")
    print(f"[OK] Test Accuracy: {results[1]:.4f}")
    if len(results) > 2:
        print(f"[OK] Test Top-3 Accuracy: {results[2]:.4f}")
    
    return results


def generate_predictions(model, test_gen):
    """Generate predictions for confusion matrix."""
    print("\n[>>] Generating predictions...")
    
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    return y_true, y_pred, predictions


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Plot and save confusion matrix."""
    print("\n[>>] Creating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def save_classification_report(y_true, y_pred, class_names, output_dir):
    """Save detailed classification report."""
    print("\n[>>] Creating classification report...")
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(report)
    
    print(f"[OK] Saved: {report_path}")
    
    # Also print to console
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(report)


def plot_per_class_accuracy(y_true, y_pred, class_names, output_dir):
    """Plot per-class accuracy."""
    print("\n[>>] Creating per-class accuracy plot...")
    
    # Calculate per-class accuracy
    accuracies = []
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
            accuracies.append(class_acc)
        else:
            accuracies.append(0)
    
    # Plot
    plt.figure(figsize=(16, 6))
    colors = ['green' if acc >= 0.9 else 'orange' if acc >= 0.7 else 'red' for acc in accuracies]
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
    
    output_path = output_dir / 'per_class_accuracy.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def save_evaluation_summary(results, y_true, y_pred, class_names, output_dir):
    """Save evaluation summary as JSON."""
    print("\n[>>] Saving evaluation summary...")
    
    # Calculate metrics
    accuracy = (y_true == y_pred).sum() / len(y_true)
    
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
            per_class_acc[class_name] = float(class_acc)
    
    summary = {
        'test_loss': float(results[0]),
        'test_accuracy': float(results[1]),
        'test_top3_accuracy': float(results[2]) if len(results) > 2 else None,
        'overall_accuracy': float(accuracy),
        'num_samples': int(len(y_true)),
        'num_classes': int(len(class_names)),
        'per_class_accuracy': per_class_acc
    }
    
    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"[OK] Saved: {summary_path}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("STEP 4: EVALUATE SIGN LANGUAGE RECOGNITION MODEL")
    print("="*70)
    
    # Define paths
    model_path = project_root / 'models' / 'saved_models' / 'best_model.keras'
    data_dir = project_root / 'data' / 'raw' / 'asl_alphabet_train' / 'asl_alphabet_train'
    output_dir = project_root / 'outputs' / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Create test generator (using validation split as test)
    test_gen = create_test_generator(data_dir, img_size=(224, 224), batch_size=32)
    class_names = list(test_gen.class_indices.keys())
    
    # Evaluate model
    results = evaluate_model(model, test_gen)
    
    # Generate predictions
    y_true, y_pred, predictions = generate_predictions(model, test_gen)
    
    # Create visualizations and reports
    plot_confusion_matrix(y_true, y_pred, class_names, output_dir)
    save_classification_report(y_true, y_pred, class_names, output_dir)
    plot_per_class_accuracy(y_true, y_pred, class_names, output_dir)
    save_evaluation_summary(results, y_true, y_pred, class_names, output_dir)
    
    print("\n" + "="*70)
    print("[OK] STEP 4 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n[>>] Evaluation outputs saved to: {output_dir}")
    print("\nNext step: Real-time detection")
    print("Command: python scripts/step5_realtime_detection.py")


if __name__ == "__main__":
    main()
