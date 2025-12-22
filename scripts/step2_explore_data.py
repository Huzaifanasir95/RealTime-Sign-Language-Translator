"""
Step 2: Explore and Analyze ASL Alphabet Dataset

This script explores the dataset, generates statistics, and creates visualizations.

Usage:
    conda activate timegan-gpu
    python scripts/step2_explore_data.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def check_dataset_exists():
    """Check if dataset has been downloaded."""
    print("\n" + "="*70)
    print("CHECKING DATASET")
    print("="*70 + "\n")
    
    data_dir = project_root / 'data' / 'raw' / 'asl_alphabet_train' / 'asl_alphabet_train'
    
    if not data_dir.exists():
        print(f"[X] Dataset not found at: {data_dir}")
        print("\nPlease run the download script first:")
        print("  python scripts/step1_download_dataset.py")
        return None
    
    print(f"[OK] Dataset found at: {data_dir}")
    return data_dir


def explore_dataset(data_dir):
    """Explore dataset structure and statistics."""
    print("\n" + "="*70)
    print("DATASET EXPLORATION")
    print("="*70 + "\n")
    
    # Get class names
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    num_classes = len(class_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {', '.join(class_names)}\n")
    
    # Count images per class
    class_counts = {}
    for class_name in class_names:
        class_path = data_dir / class_name
        count = len(list(class_path.glob('*.jpg'))) + len(list(class_path.glob('*.png')))
        class_counts[class_name] = count
    
    # Create DataFrame
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    df = df.sort_values('Count', ascending=False)
    
    print("Images per class:")
    print(df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("STATISTICS")
    print("="*70)
    print(f"  Total images: {df['Count'].sum():,}")
    print(f"  Average per class: {df['Count'].mean():.0f}")
    print(f"  Min images: {df['Count'].min()}")
    print(f"  Max images: {df['Count'].max()}")
    print(f"  Std deviation: {df['Count'].std():.2f}")
    
    return df, class_names


def visualize_class_distribution(df, output_dir):
    """Create and save class distribution plot."""
    print("\n[>>] Creating class distribution plot...")
    
    plt.figure(figsize=(16, 6))
    plt.bar(df['Class'], df['Count'], color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
    plt.title('ASL Alphabet Dataset - Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'class_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def visualize_sample_images(data_dir, class_names, output_dir):
    """Create and save sample images visualization."""
    print("[>>] Creating sample images visualization...")
    
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    axes = axes.ravel()
    
    for idx, class_name in enumerate(class_names[:30]):
        class_path = data_dir / class_name
        # Get first image
        img_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        if img_files:
            img_path = img_files[0]
            img = Image.open(img_path)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"{class_name}", fontsize=10, fontweight='bold')
            axes[idx].axis('off')
    
    plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'sample_images.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def visualize_class_variations(data_dir, class_name, output_dir):
    """Visualize variations within a single class."""
    print(f"[>>] Creating variations visualization for class '{class_name}'...")
    
    class_path = data_dir / class_name
    img_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
    img_files = img_files[:12]  # Take first 12 images
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    
    for idx, img_path in enumerate(img_files):
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
    
    plt.suptitle(f'Variations in Class "{class_name}"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / f'variations_class_{class_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def analyze_image_properties(data_dir, class_names):
    """Analyze image properties (dimensions, data type, etc.)."""
    print("\n" + "="*70)
    print("IMAGE PROPERTIES ANALYSIS")
    print("="*70 + "\n")
    
    sample_images = []
    for class_name in class_names[:5]:
        class_path = data_dir / class_name
        img_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        if img_files:
            img_path = img_files[0]
            img = cv2.imread(str(img_path))
            sample_images.append((class_name, img))
    
    print("Image dimensions:")
    for class_name, img in sample_images:
        print(f"  Class {class_name}: {img.shape}")
    
    if sample_images:
        _, first_img = sample_images[0]
        print(f"\nData type: {first_img.dtype}")
        print(f"Pixel value range: [{first_img.min()}, {first_img.max()}]")
        print(f"Mean pixel value: {first_img.mean():.2f}")


def save_summary_report(df, class_names, output_dir):
    """Save a text summary report."""
    print("\n[>>] Creating summary report...")
    
    report_path = output_dir / 'data_exploration_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ASL ALPHABET DATASET - EXPLORATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Classes: {len(class_names)}\n")
        f.write(f"Total Images: {df['Count'].sum():,}\n\n")
        
        f.write("Class Names:\n")
        f.write(f"{', '.join(class_names)}\n\n")
        
        f.write("Statistics:\n")
        f.write(f"  Average images per class: {df['Count'].mean():.0f}\n")
        f.write(f"  Min images: {df['Count'].min()}\n")
        f.write(f"  Max images: {df['Count'].max()}\n")
        f.write(f"  Std deviation: {df['Count'].std():.2f}\n\n")
        
        f.write("Images per class:\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("Next Steps:\n")
        f.write("="*70 + "\n")
        f.write("1. Data preprocessing and augmentation\n")
        f.write("2. Model architecture selection\n")
        f.write("3. Model training\n")
        f.write("4. Model evaluation\n")
        f.write("5. Real-time detection system\n")
    
    print(f"[OK] Saved: {report_path}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("STEP 2: EXPLORE ASL ALPHABET DATASET")
    print("="*70)
    
    # Check if dataset exists
    data_dir = check_dataset_exists()
    if data_dir is None:
        return
    
    # Create output directory
    output_dir = project_root / 'outputs' / 'exploration'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Explore dataset
    df, class_names = explore_dataset(data_dir)
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    visualize_class_distribution(df, output_dir)
    visualize_sample_images(data_dir, class_names, output_dir)
    visualize_class_variations(data_dir, 'A', output_dir)  # Example: class 'A'
    
    # Analyze properties
    analyze_image_properties(data_dir, class_names)
    
    # Save summary report
    save_summary_report(df, class_names, output_dir)
    
    print("\n" + "="*70)
    print("[OK] STEP 2 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n[>>] Outputs saved to: {output_dir}")
    print("\nNext step: Model training")
    print("Command: python scripts/step3_train_model.py")


if __name__ == "__main__":
    main()
