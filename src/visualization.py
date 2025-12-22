"""
Visualization utilities for data exploration and results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd


def plot_class_distribution(class_counts: dict,
                            output_path: Path,
                            figsize: Tuple[int, int] = (16, 6)):
    """
    Plot class distribution bar chart.
    
    Args:
        class_counts: Dictionary mapping class name to count
        output_path: Path to save plot
        figsize: Figure size
    """
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    df = df.sort_values('Count', ascending=False)
    
    plt.figure(figsize=figsize)
    plt.bar(df['Class'], df['Count'], color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
    plt.title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (class_name, count) in enumerate(zip(df['Class'], df['Count'])):
        plt.text(i, count, str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sample_images(data_dir: Path,
                       class_names: List[str],
                       output_path: Path,
                       samples_per_class: int = 1,
                       figsize: Tuple[int, int] = (18, 15)):
    """
    Plot sample images from each class.
    
    Args:
        data_dir: Directory containing class subdirectories
        class_names: List of class names
        output_path: Path to save plot
        samples_per_class: Number of samples per class
        figsize: Figure size
    """
    n_classes = len(class_names)
    n_cols = 6
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel()
    
    for idx, class_name in enumerate(class_names):
        class_path = data_dir / class_name
        img_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        
        if img_files:
            img_path = img_files[0]
            img = Image.open(img_path)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"{class_name}", fontsize=10, fontweight='bold')
            axes[idx].axis('off')
        else:
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_classes, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_class_variations(data_dir: Path,
                         class_name: str,
                         output_path: Path,
                         n_samples: int = 12,
                         figsize: Tuple[int, int] = (12, 9)):
    """
    Plot variations within a single class.
    
    Args:
        data_dir: Directory containing class subdirectories
        class_name: Class name to visualize
        output_path: Path to save plot
        n_samples: Number of samples to show
        figsize: Figure size
    """
    class_path = data_dir / class_name
    img_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
    img_files = img_files[:n_samples]
    
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel()
    
    for idx, img_path in enumerate(img_files):
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(img_files), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Variations in Class "{class_name}"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_image_properties(images: List[np.ndarray],
                          class_names: List[str],
                          output_path: Path,
                          figsize: Tuple[int, int] = (15, 10)):
    """
    Plot image property distributions.
    
    Args:
        images: List of images
        class_names: Corresponding class names
        output_path: Path to save plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Collect statistics
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    means = [img.mean() for img in images]
    stds = [img.std() for img in images]
    
    # Plot height distribution
    axes[0, 0].hist(heights, bins=20, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Image Height Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Height (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot width distribution
    axes[0, 1].hist(widths, bins=20, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Image Width Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Width (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot mean pixel value distribution
    axes[1, 0].hist(means, bins=20, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Mean Pixel Value Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Mean Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot std pixel value distribution
    axes[1, 1].hist(stds, bins=20, color='plum', edgecolor='black')
    axes[1, 1].set_title('Pixel Value Std Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Std Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_augmentation_examples(original_img: np.ndarray,
                               augmented_images: List[np.ndarray],
                               output_path: Path,
                               figsize: Tuple[int, int] = (15, 10)):
    """
    Plot original and augmented image examples.
    
    Args:
        original_img: Original image
        augmented_images: List of augmented versions
        output_path: Path to save plot
        figsize: Figure size
    """
    n_images = len(augmented_images) + 1
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel()
    
    # Plot original
    axes[0].imshow(original_img)
    axes[0].set_title('Original', fontweight='bold', color='green')
    axes[0].axis('off')
    
    # Plot augmented
    for idx, aug_img in enumerate(augmented_images):
        axes[idx + 1].imshow(aug_img)
        axes[idx + 1].set_title(f'Augmented {idx + 1}', fontweight='bold')
        axes[idx + 1].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_prediction_visualization(image: np.ndarray,
                                    true_label: str,
                                    pred_label: str,
                                    confidence: float,
                                    top_k_preds: List[Tuple[str, float]],
                                    output_path: Path,
                                    figsize: Tuple[int, int] = (12, 6)):
    """
    Create visualization for a single prediction.
    
    Args:
        image: Input image
        true_label: True class label
        pred_label: Predicted class label
        confidence: Prediction confidence
        top_k_preds: List of (label, confidence) for top-k predictions
        output_path: Path to save plot
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot image
    ax1.imshow(image)
    title_color = 'green' if true_label == pred_label else 'red'
    ax1.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2%})',
                  fontweight='bold', color=title_color, fontsize=12)
    ax1.axis('off')
    
    # Plot top-k predictions
    labels = [label for label, _ in top_k_preds]
    confidences = [conf for _, conf in top_k_preds]
    colors = ['green' if label == true_label else 'steelblue' for label in labels]
    
    ax2.barh(labels, confidences, color=colors, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Confidence', fontweight='bold')
    ax2.set_title('Top Predictions', fontweight='bold', fontsize=12)
    ax2.set_xlim([0, 1])
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
